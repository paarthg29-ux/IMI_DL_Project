import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
#  CVAE v2 — fixes vs v1:                                            #
#  1. Condition injected at EVERY hidden layer (not just input)      #
#  2. Latent dim 16 → 32  (more room to encode chemical diversity)   #
#  3. Scheduler patience 20 → 40  (LR was dying too fast before)     #
#  4. Condition gets its own learned embedding (COND_DIM=16)         #
# ------------------------------------------------------------------ #

LATENT_DIM    = 32
EMBED_DIM     = 64
HIDDEN_DIM    = 256
COND_DIM      = 16    # condition projected to this dim before injection
EPOCHS        = 300
BATCH_SIZE    = 64
LR            = 1e-3
BETA_MAX      = 1.0
ANNEAL_EPOCHS = 60

# --- 1. LOAD EMBEDDINGS ---
print("Loading embeddings...")
data       = torch.load("embeddings.pt")
embeddings = data["embeddings"]            # (N, 64)
targets    = data["targets"].unsqueeze(1)  # (N, 1) real GPa

# Normalise condition
log_targets = torch.log1p(targets)
cond_mean   = log_targets.mean()
cond_std    = log_targets.std()
conditions  = (log_targets - cond_mean) / (cond_std + 1e-8)

# Normalise embeddings
emb_mean        = embeddings.mean(dim=0, keepdim=True)
emb_std         = embeddings.std(dim=0,  keepdim=True)
embeddings_norm = (embeddings - emb_mean) / (emb_std + 1e-8)

print(f"Dataset: {embeddings.shape[0]} crystals | "
      f"GPa range {targets.min():.0f}–{targets.max():.0f}")

torch.save({
    "emb_mean": emb_mean, "emb_std": emb_std,
    "cond_mean": cond_mean, "cond_std": cond_std
}, "cvae_norm_stats.pt")

# --- 2. TRAIN / VAL SPLIT ---
dataset  = TensorDataset(embeddings_norm, conditions)
n_val    = int(0.15 * len(dataset))
n_train  = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                 generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# --- 3. CVAE ARCHITECTURE v2 ---
class ConditionEmbedder(nn.Module):
    """
    FIX 1: Give the condition its own learned embedding.
    A scalar GPa value is too weak when concatenated to a 256-dim vector.
    This projects it to COND_DIM=16 before injection.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, COND_DIM),
        )
    def forward(self, cond):
        return self.net(cond)   # (B, COND_DIM)


class Encoder(nn.Module):
    """
    FIX 1 (cont): Condition injected at input AND after each hidden layer.
    This forces every layer to be aware of the target GPa.
    """
    def __init__(self):
        super().__init__()
        self.cond_embedder = ConditionEmbedder()

        self.fc1 = nn.Linear(EMBED_DIM + COND_DIM, HIDDEN_DIM)
        self.ln1 = nn.LayerNorm(HIDDEN_DIM)

        self.fc2 = nn.Linear(HIDDEN_DIM + COND_DIM, HIDDEN_DIM)
        self.ln2 = nn.LayerNorm(HIDDEN_DIM)

        self.mu_head      = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.log_var_head = nn.Linear(HIDDEN_DIM, LATENT_DIM)

    def forward(self, emb, cond):
        c  = self.cond_embedder(cond)

        h  = F.relu(self.ln1(self.fc1(torch.cat([emb, c], dim=1))))
        h  = F.relu(self.ln2(self.fc2(torch.cat([h,   c], dim=1))))  # re-inject

        mu      = self.mu_head(h)
        log_var = self.log_var_head(h).clamp(-4, 4)
        return mu, log_var


class Decoder(nn.Module):
    """Condition injected at every decoder layer."""
    def __init__(self):
        super().__init__()
        self.cond_embedder = ConditionEmbedder()

        self.fc1 = nn.Linear(LATENT_DIM + COND_DIM, HIDDEN_DIM)
        self.ln1 = nn.LayerNorm(HIDDEN_DIM)

        self.fc2 = nn.Linear(HIDDEN_DIM + COND_DIM, HIDDEN_DIM)
        self.ln2 = nn.LayerNorm(HIDDEN_DIM)

        self.out = nn.Linear(HIDDEN_DIM, EMBED_DIM)

    def forward(self, z, cond):
        c = self.cond_embedder(cond)

        h = F.relu(self.ln1(self.fc1(torch.cat([z, c], dim=1))))
        h = F.relu(self.ln2(self.fc2(torch.cat([h, c], dim=1))))
        return self.out(h)


class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterise(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def forward(self, emb, cond):
        mu, log_var = self.encoder(emb, cond)
        z           = self.reparameterise(mu, log_var)
        recon       = self.decoder(z, cond)
        return recon, mu, log_var

    def sample(self, cond_normalised, n_samples, device):
        """Inference: sample n_samples embeddings for one condition."""
        with torch.no_grad():
            z    = torch.randn(n_samples, LATENT_DIM).to(device)
            cond = cond_normalised.expand(n_samples, -1).to(device)
            return self.decoder(z, cond)   # (n_samples, EMBED_DIM)


# --- 4. LOSS ---
def cvae_loss(recon, target_emb, mu, log_var, beta):
    recon_loss = F.mse_loss(recon, target_emb, reduction='mean')
    kl_loss    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_loss, recon_loss.item(), kl_loss.item()


# --- 5. TRAINING SETUP ---
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = CVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# FIX 3: patience=40 so LR doesn't collapse before epoch 100
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=40
)

total_params = sum(p.numel() for p in model.parameters())
print(f"CVAE v2 | Latent={LATENT_DIM} | Hidden={HIDDEN_DIM} | "
      f"Params={total_params:,} | Device={device}")
print(f"Train: {n_train} | Val: {n_val}")

# --- 6. TRAINING LOOP ---
train_losses, val_losses = [], []
recon_log, kl_log        = [], []
best_val_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    beta = min(BETA_MAX, BETA_MAX * epoch / ANNEAL_EPOCHS)

    model.train()
    ep_loss = ep_recon = ep_kl = 0.0

    for emb_b, cond_b in train_loader:
        emb_b, cond_b = emb_b.to(device), cond_b.to(device)
        optimizer.zero_grad()
        recon, mu, lv  = model(emb_b, cond_b)
        loss, r, k     = cvae_loss(recon, emb_b, mu, lv, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        ep_loss += loss.item(); ep_recon += r; ep_kl += k

    avg_train = ep_loss  / len(train_loader)
    avg_recon = ep_recon / len(train_loader)
    avg_kl    = ep_kl    / len(train_loader)

    model.eval()
    vl = 0.0
    with torch.no_grad():
        for emb_b, cond_b in val_loader:
            emb_b, cond_b = emb_b.to(device), cond_b.to(device)
            recon, mu, lv = model(emb_b, cond_b)
            l, _, _       = cvae_loss(recon, emb_b, mu, lv, beta)
            vl           += l.item()

    avg_val = vl / len(val_loader)
    scheduler.step(avg_val)

    train_losses.append(avg_train)
    val_losses.append(avg_val)
    recon_log.append(avg_recon)
    kl_log.append(avg_kl)

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), "cvae_best.pth")

    if epoch % 25 == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:>3}/{EPOCHS} | "
              f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
              f"Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | "
              f"β: {beta:.2f} | LR: {lr:.6f}")

print(f"\n✅ Best val loss: {best_val_loss:.4f}")
torch.save({"model_state": model.state_dict(),
            "latent_dim":  LATENT_DIM,
            "embed_dim":   EMBED_DIM,
            "hidden_dim":  HIDDEN_DIM,
            "cond_dim":    COND_DIM}, "cvae_model.pth")
print("✅ Saved to 'cvae_model.pth'")

# --- 7. DIAGNOSTICS ---
print("Generating diagnostics...")
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

axs[0].plot(train_losses, label='Train', color='royalblue', lw=2)
axs[0].plot(val_losses,   label='Val',   color='tomato',    lw=2)
axs[0].set_title('Total Loss (Train vs Val)')
axs[0].set_xlabel('Epoch'); axs[0].set_ylabel('Loss')
axs[0].legend(); axs[0].grid(True, linestyle='--', alpha=0.6)

axs[1].plot(recon_log, label='Reconstruction', color='green',  lw=2)
axs[1].plot(kl_log,    label='KL Divergence',  color='orange', lw=2)
axs[1].set_title('Recon vs KL Loss')
axs[1].set_xlabel('Epoch'); axs[1].set_ylabel('Loss component')
axs[1].legend(); axs[1].grid(True, linestyle='--', alpha=0.6)

# PCA cluster plot
model.load_state_dict(torch.load("cvae_best.pth", map_location=device))
model.eval()

from sklearn.decomposition import PCA
gpa_bins = [50, 100, 150, 200, 300]
colors   = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
all_dec  = []
all_lbl  = []

with torch.no_grad():
    for gpa in gpa_bins:
        log_c   = torch.log1p(torch.tensor([[float(gpa)]])).to(device)
        norm_c  = (log_c - cond_mean.to(device)) / (cond_std.to(device) + 1e-8)
        decoded = model.sample(norm_c, n_samples=100, device=device).cpu().numpy()
        all_dec.append(decoded)
        all_lbl.extend([gpa] * 100)

all_dec = np.vstack(all_dec)
pca     = PCA(n_components=2)
z_2d    = pca.fit_transform(all_dec)

for i, gpa in enumerate(gpa_bins):
    mask = np.array(all_lbl) == gpa
    axs[2].scatter(z_2d[mask,0], z_2d[mask,1],
                   label=f'{gpa} GPa', color=colors[i], alpha=0.5, s=20)

axs[2].set_title('Decoded Embedding Space (PCA)\nColoured by target GPa')
axs[2].set_xlabel('PC 1'); axs[2].set_ylabel('PC 2')
axs[2].legend(title='Target', fontsize=8)
axs[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("cvae_diagnostics.png", dpi=300)
print("✅ Saved 'cvae_diagnostics.png'")
print("\nNext step: run generate_candidates.py")