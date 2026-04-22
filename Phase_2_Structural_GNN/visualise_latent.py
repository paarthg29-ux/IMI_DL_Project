import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
#  visualise_latent.py                                               #
#                                                                    #
#  Encodes all 1497 real crystal embeddings into the CVAE latent     #
#  space (mu vectors), then plots them in 2D using UMAP.             #
#                                                                    #
#  What to look for:                                                 #
#  - A smooth colour gradient (low → high GPa) = good conditioning  #
#  - Hard clusters by GPa bin = excellent conditioning               #
#  - Random salt-and-pepper colour = conditioning isn't working      #
# ------------------------------------------------------------------ #

# Install umap if needed: pip install umap-learn
try:
    import umap
except ImportError:
    print("Installing umap-learn...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "umap-learn", "--quiet"])
    import umap

LATENT_DIM = 32
EMBED_DIM  = 64
HIDDEN_DIM = 256
COND_DIM   = 16

# --- 1. REBUILD CVAE (must match train_cvae.py) ---
class ConditionEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, COND_DIM),
        )
    def forward(self, cond):
        return self.net(cond)

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
        c = self.cond_embedder(cond)
        h = F.relu(self.ln1(self.fc1(torch.cat([emb, c], dim=1))))
        h = F.relu(self.ln2(self.fc2(torch.cat([h,   c], dim=1))))
        return self.mu_head(h), self.log_var_head(h).clamp(-4, 4)

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

class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def encode_mu(self, emb, cond):
        """Return only mu (the deterministic centre of the posterior)."""
        mu, _ = self.encoder(emb, cond)
        return mu

    def sample(self, cond_norm, n_samples, device):
        with torch.no_grad():
            z    = torch.randn(n_samples, LATENT_DIM).to(device)
            cond = cond_norm.expand(n_samples, -1).to(device)
            return self.decoder(z, cond)


# --- 2. LOAD EVERYTHING ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cvae = CVAE().to(device)
ckpt = torch.load("cvae_model.pth", map_location=device)
cvae.load_state_dict(ckpt["model_state"])
cvae.eval()
print("✅ Loaded CVAE")

data      = torch.load("embeddings.pt")
embeddings = data["embeddings"].to(device)   # (N, 64) — raw GNN embeddings
targets    = data["targets"]                 # (N,)    — real GPa

stats     = torch.load("cvae_norm_stats.pt", map_location=device)
emb_mean  = stats["emb_mean"].to(device)
emb_std   = stats["emb_std"].to(device)
cond_mean = stats["cond_mean"].to(device)
cond_std  = stats["cond_std"].to(device)

# Normalise embeddings and conditions
emb_norm  = (embeddings - emb_mean) / (emb_std + 1e-8)
log_cond  = torch.log1p(targets.unsqueeze(1).to(device))
cond_norm = (log_cond - cond_mean) / (cond_std + 1e-8)

# --- 3. ENCODE ALL CRYSTALS → LATENT MU VECTORS ---
print(f"Encoding {len(embeddings)} crystals into latent space...")
with torch.no_grad():
    mu_vectors = cvae.encode_mu(emb_norm, cond_norm)  # (N, LATENT_DIM)

mu_np  = mu_vectors.cpu().numpy()
gpa_np = targets.numpy()

# --- 4. UMAP REDUCTION: 32-dim → 2-dim ---
print("Running UMAP (this takes ~30 seconds)...")
reducer  = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1,
                     metric='euclidean', random_state=42)
z_2d     = reducer.fit_transform(mu_np)   # (N, 2)
print("✅ UMAP complete")

# --- 5. ALSO PROJECT GENERATED SAMPLES FOR 5 TARGET GPa VALUES ---
gen_gpa_bins  = [50, 100, 150, 200, 300]
gen_markers   = ['*', 'D', '^', 's', 'P']
gen_colors    = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
gen_points_2d = []

print("Projecting generated samples into UMAP space...")
with torch.no_grad():
    for gpa in gen_gpa_bins:
        lc      = torch.log1p(torch.tensor([[float(gpa)]])).to(device)
        nc      = (lc - cond_mean) / (cond_std + 1e-8)
        samples = cvae.sample(nc, n_samples=50, device=device)  # (50, 64)

        # Encode generated embeddings back through encoder to get mu
        # (need the condition for that — use the same target GPa)
        nc_batch = nc.expand(50, -1)
        mu_gen   = cvae.encode_mu(samples, nc_batch)
        gen_points_2d.append(mu_gen.cpu().numpy())

gen_points_2d = [reducer.transform(g) for g in gen_points_2d]

# --- 6. PLOT ---
fig, axs = plt.subplots(1, 2, figsize=(18, 7))

# --- Plot 1: All real crystals, coloured continuously by GPa ---
norm    = Normalize(vmin=gpa_np.min(), vmax=np.percentile(gpa_np, 95))
cmap    = cm.plasma
scatter = axs[0].scatter(z_2d[:, 0], z_2d[:, 1],
                          c=gpa_np, cmap=cmap, norm=norm,
                          s=18, alpha=0.75, edgecolors='none')
cbar = plt.colorbar(scatter, ax=axs[0])
cbar.set_label('Bulk Modulus (GPa)', fontsize=11)
axs[0].set_title('UMAP of Latent Space\nReal crystals coloured by bulk modulus', fontsize=12)
axs[0].set_xlabel('UMAP 1'); axs[0].set_ylabel('UMAP 2')
axs[0].grid(True, linestyle='--', alpha=0.4)

# --- Plot 2: Real crystals (grey) + generated samples (coloured stars) ---
axs[1].scatter(z_2d[:, 0], z_2d[:, 1],
               c='lightgrey', s=12, alpha=0.5, edgecolors='none', label='Real crystals')

for i, (gpa, pts) in enumerate(zip(gen_gpa_bins, gen_points_2d)):
    axs[1].scatter(pts[:, 0], pts[:, 1],
                   color=gen_colors[i], marker=gen_markers[i],
                   s=80, alpha=0.9, edgecolors='black', linewidths=0.5,
                   label=f'Generated @ {gpa} GPa')

axs[1].set_title('UMAP: Real vs Generated Crystals\nColoured by target GPa', fontsize=12)
axs[1].set_xlabel('UMAP 1'); axs[1].set_ylabel('UMAP 2')
axs[1].legend(fontsize=9, loc='best')
axs[1].grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig("latent_space_umap.png", dpi=300)
print("✅ Saved 'latent_space_umap.png'")

# --- 7. BONUS: GPa bin statistics in latent space ---
print("\n--- LATENT SPACE STATISTICS BY GPa BIN ---")
bins   = [(0,50), (50,100), (100,150), (150,200), (200,400)]
labels = ['0-50', '50-100', '100-150', '150-200', '200+']

for (lo, hi), label in zip(bins, labels):
    mask = (gpa_np >= lo) & (gpa_np < hi)
    if mask.sum() == 0:
        continue
    pts  = z_2d[mask]
    print(f"  {label:>10} GPa | n={mask.sum():>4} | "
          f"UMAP centroid = ({pts[:,0].mean():+.2f}, {pts[:,1].mean():+.2f})")
