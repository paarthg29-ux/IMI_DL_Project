import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import CGConv, global_mean_pool
from torch_geometric.loader import DataLoader
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
#  This script loads your TRAINED GNN, removes the prediction head,  #
#  and uses the remaining layers as a feature extractor.             #
#  Output: embeddings.pt — a dict with:                              #
#    "embeddings" → (N, 64) float tensor                             #
#    "targets"    → (N,)    float tensor  [real GPa, not log-space]  #
# ------------------------------------------------------------------ #

# --- 1. REBUILD THE EXACT SAME ARCHITECTURE FROM train_gnn.py ---
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

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.bn0(self.input_proj(x)))
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_embedding(self, x, edge_index, edge_attr, batch):
        """Same forward pass but STOPS before the prediction head.
           Returns the 64-dim crystal fingerprint."""
        x = F.relu(self.bn0(self.input_proj(x)))
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)   # ← 64-dim fingerprint lives here
        return x                          #   (before fc1 / fc2)

# --- 2. LOAD WEIGHTS ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = CrystalGNN().to(device)
model.load_state_dict(torch.load("crystal_gnn_model.pth", map_location=device))
model.eval()
print(f"✅ Loaded trained GNN weights onto: {device}")

# --- 3. LOAD THE FULL DATASET (un-transformed targets) ---
raw_dataset = torch.load("crystal_graphs_dataset.pt", weights_only=False)

# The dataset was saved with RAW GPa values by build_graphs.py.
# We log-transform here only for the GNN forward pass consistency check,
# but we STORE the real GPa values for the CVAE condition.
loader = DataLoader(raw_dataset, batch_size=64, shuffle=False)

all_embeddings = []
all_targets    = []   # real GPa

print(f"Extracting embeddings from {len(raw_dataset)} crystal graphs...")

with torch.no_grad():
    for data in loader:
        data           = data.to(device)
        data.x         = torch.nan_to_num(data.x,         nan=0.0)
        data.edge_attr = torch.nan_to_num(data.edge_attr, nan=0.0)

        emb = model.get_embedding(data.x, data.edge_index, data.edge_attr, data.batch)
        all_embeddings.append(emb.cpu())

        # data.y is raw GPa from build_graphs.py
        all_targets.append(data.y.cpu().squeeze())

embeddings = torch.cat(all_embeddings, dim=0)   # (N, 64)
targets    = torch.cat(all_targets,    dim=0)   # (N,)

print(f"\n--- EXTRACTION COMPLETE ---")
print(f"Embedding matrix shape : {embeddings.shape}")
print(f"Target vector shape    : {targets.shape}")
print(f"Bulk modulus range     : {targets.min():.1f} – {targets.max():.1f} GPa")
print(f"Mean bulk modulus      : {targets.mean():.1f} GPa")

# --- 4. SAVE ---
torch.save({"embeddings": embeddings, "targets": targets}, "embeddings.pt")
print("\n✅ Saved to 'embeddings.pt'  →  ready for train_cvae.py")