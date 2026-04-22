import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import CGConv, global_mean_pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
#  generate_candidates.py                                            #
#                                                                    #
#  1. Load trained CVAE + norm stats                                 #
#  2. Sample N synthetic embeddings at a user-defined target GPa     #
#  3. De-normalise embeddings back to GNN space                      #
#  4. Pass through GNN prediction head → get predicted bulk modulus  #
#  5. Rank by proximity to target, save results                      #
# ------------------------------------------------------------------ #

# ---- USER SETTINGS ------------------------------------------------ #
TARGET_GPA   = 200.0   # ← change this to whatever you want to generate
N_SAMPLES    = 500     # how many candidates to generate
TOP_K        = 20      # how many to show in the final ranked table
# ------------------------------------------------------------------- #

LATENT_DIM = 32
EMBED_DIM  = 64
HIDDEN_DIM = 256
COND_DIM   = 16

# --- 1. REBUILD CVAE ARCHITECTURE (must match train_cvae.py) ---
class ConditionEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, COND_DIM),
        )
    def forward(self, cond):
        return self.net(cond)

class Decoder(nn.Module):
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

class Encoder(nn.Module):
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
        h  = F.relu(self.ln2(self.fc2(torch.cat([h,   c], dim=1))))
        return self.mu_head(h), self.log_var_head(h).clamp(-4, 4)

class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterise(self, mu, log_var):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

    def forward(self, emb, cond):
        mu, lv = self.encoder(emb, cond)
        return self.decoder(self.reparameterise(mu, lv), cond), mu, lv

    def sample(self, cond_norm, n_samples, device):
        with torch.no_grad():
            z    = torch.randn(n_samples, LATENT_DIM).to(device)
            cond = cond_norm.expand(n_samples, -1).to(device)
            return self.decoder(z, cond)


# --- 2. REBUILD GNN (must match train_gnn.py) ---
class CrystalGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = Linear(12, 64)
        self.bn0   = torch.nn.BatchNorm1d(64)
        self.conv1 = CGConv(channels=64, dim=1, batch_norm=True)
        self.conv2 = CGConv(channels=64, dim=1, batch_norm=True)
        self.conv3 = CGConv(channels=64, dim=1, batch_norm=True)
        self.fc1   = Linear(64, 32)
        self.fc2   = Linear(32, 1)

    def predict_from_embedding(self, emb):
        """
        Skip the graph conv layers — take a 64-dim embedding directly
        into the prediction head. This is how we score CVAE outputs.
        """
        x = F.relu(self.fc1(emb))
        return self.fc2(x)


# --- 3. LOAD EVERYTHING ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load CVAE
cvae = CVAE().to(device)
ckpt = torch.load("cvae_model.pth", map_location=device)
cvae.load_state_dict(ckpt["model_state"])
cvae.eval()
print("✅ Loaded CVAE")

# Load GNN (only need fc1 + fc2 prediction head)
gnn = CrystalGNN().to(device)
gnn.load_state_dict(torch.load("crystal_gnn_model.pth", map_location=device))
gnn.eval()
print("✅ Loaded GNN scorer")

# Load normalisation stats
stats     = torch.load("cvae_norm_stats.pt", map_location=device)
emb_mean  = stats["emb_mean"].to(device)
emb_std   = stats["emb_std"].to(device)
cond_mean = stats["cond_mean"].to(device)
cond_std  = stats["cond_std"].to(device)
print("✅ Loaded normalisation stats")

# --- 4. ENCODE THE TARGET CONDITION ---
log_target = torch.log1p(torch.tensor([[TARGET_GPA]])).to(device)
norm_cond  = (log_target - cond_mean) / (cond_std + 1e-8)  # (1, 1)

print(f"\nGenerating {N_SAMPLES} candidates at target = {TARGET_GPA} GPa...")

# --- 5. SAMPLE FROM CVAE ---
with torch.no_grad():
    # (N_SAMPLES, EMBED_DIM) — still in normalised embedding space
    synth_emb_norm = cvae.sample(norm_cond, N_SAMPLES, device)

    # De-normalise back to real GNN embedding space
    synth_emb = synth_emb_norm * (emb_std + 1e-8) + emb_mean  # (N_SAMPLES, 64)

    # --- 6. SCORE WITH GNN PREDICTION HEAD ---
    # The GNN was trained on log-transformed targets, so output is log-space
    log_pred = gnn.predict_from_embedding(synth_emb)           # (N_SAMPLES, 1)
    pred_gpa = torch.expm1(log_pred).squeeze()                 # back to real GPa

pred_gpa_np = pred_gpa.cpu().numpy()

# --- 7. RANK CANDIDATES ---
# Score = absolute distance from target GPa (lower = better)
distances = np.abs(pred_gpa_np - TARGET_GPA)
ranked_idx = np.argsort(distances)

print(f"\n--- TOP {TOP_K} CANDIDATES (closest to {TARGET_GPA} GPa) ---")
print(f"{'Rank':<6} {'Predicted GPa':<16} {'Distance from Target':<22}")
print("-" * 46)

top_indices   = ranked_idx[:TOP_K]
top_pred_gpa  = pred_gpa_np[top_indices]
top_distances = distances[top_indices]

for rank, (idx, gpa, dist) in enumerate(zip(top_indices, top_pred_gpa, top_distances), 1):
    print(f"{rank:<6} {gpa:<16.2f} {dist:<22.2f}")

# Save full results to CSV
results_df = pd.DataFrame({
    "candidate_id":    range(N_SAMPLES),
    "predicted_gpa":   pred_gpa_np,
    "distance_to_target": distances,
})
results_df = results_df.sort_values("distance_to_target").reset_index(drop=True)
results_df.to_csv(f"candidates_{int(TARGET_GPA)}gpa.csv", index=False)
print(f"\n✅ Saved all {N_SAMPLES} candidates to 'candidates_{int(TARGET_GPA)}gpa.csv'")

# --- 8. DIAGNOSTICS PLOT ---
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Distribution of predicted GPa across all candidates
axs[0].hist(pred_gpa_np, bins=40, color='steelblue', edgecolor='black', alpha=0.8)
axs[0].axvline(TARGET_GPA, color='red', lw=2, linestyle='--', label=f'Target: {TARGET_GPA} GPa')
axs[0].axvline(np.median(pred_gpa_np), color='orange', lw=2, linestyle='--',
               label=f'Median: {np.median(pred_gpa_np):.1f} GPa')
axs[0].set_title(f'Predicted GPa Distribution\n({N_SAMPLES} candidates)')
axs[0].set_xlabel('Predicted Bulk Modulus (GPa)')
axs[0].set_ylabel('Frequency')
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.6)

# Plot 2: Top-K ranked candidates
axs[1].barh(range(TOP_K, 0, -1), top_pred_gpa, color='teal', alpha=0.8)
axs[1].axvline(TARGET_GPA, color='red', lw=2, linestyle='--', label=f'Target: {TARGET_GPA} GPa')
axs[1].set_title(f'Top {TOP_K} Candidates')
axs[1].set_xlabel('Predicted Bulk Modulus (GPa)')
axs[1].set_ylabel('Rank')
axs[1].legend()
axs[1].grid(True, linestyle='--', alpha=0.6)

# Plot 3: Distance distribution (how tight is the generation?)
axs[2].hist(top_distances, bins=15, color='coral', edgecolor='black', alpha=0.8)
axs[2].set_title(f'Error Distribution — Top {TOP_K}\nMean error: {top_distances.mean():.1f} GPa')
axs[2].set_xlabel('|Predicted − Target| (GPa)')
axs[2].set_ylabel('Frequency')
axs[2].grid(True, linestyle='--', alpha=0.6)

plt.suptitle(f'CVAE Generation Results — Target: {TARGET_GPA} GPa', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"generation_results_{int(TARGET_GPA)}gpa.png", dpi=300)
print(f"✅ Saved plot to 'generation_results_{int(TARGET_GPA)}gpa.png'")
print("\nNext step: run visualise_latent.py")
