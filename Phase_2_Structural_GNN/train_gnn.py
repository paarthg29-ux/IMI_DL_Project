import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import CGConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# --- 1. LOAD THE DATA ---
print("Loading 12-Trait 3D Graph Dataset...")
dataset = torch.load("crystal_graphs_dataset.pt", weights_only=False)

# ✅ FIX 3: LOG-TRANSFORM TARGETS
# Bulk modulus is heavily right-skewed (most < 150 GPa, few up to 400+)
# Log-transform forces the model to learn hard AND soft materials equally
for graph in dataset:
    graph.y = torch.log1p(graph.y)  # log(1 + y), safe for y=0

train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# --- 2. BUILD THE NEURAL NETWORK ARCHITECTURE ---
class CrystalGNN(torch.nn.Module):
    def __init__(self):
        super(CrystalGNN, self).__init__()

        # ✅ FIX 1: CGConv ACTUALLY USES EDGE FEATURES (bond distances)
        # GCNConv was silently ignoring edge_attr entirely.
        # CGConv is designed for crystals and ingests both node + edge features.
        # dim=1 matches our 1D edge feature (bond distance in Angstroms)
        self.input_proj = Linear(12, 64)  # Project 12 node features → 64 dims first
        self.bn0 = torch.nn.BatchNorm1d(64)

        self.conv1 = CGConv(channels=64, dim=1, batch_norm=True)
        self.conv2 = CGConv(channels=64, dim=1, batch_norm=True)
        self.conv3 = CGConv(channels=64, dim=1, batch_norm=True)

        self.fc1 = Linear(64, 32)
        self.fc2 = Linear(32, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        # Project input features
        x = F.relu(self.bn0(self.input_proj(x)))

        # CGConv message passing — now using bond distances in every layer
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)

        # Global pooling: compress all atoms into 1 crystal fingerprint
        x = global_mean_pool(x, batch)

        # Final prediction head
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 3. SETUP THE TRAINING ENGINE ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CrystalGNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = torch.nn.MSELoss()

# ✅ FIX 2: LEARNING RATE SCHEDULER
# Halves the LR when loss stops improving for 15 epochs.
# Fixes the loss zigzagging you saw after epoch 60.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=15, verbose=True
)

print(f"Neural Network Built! Training on: {device}")

# --- 4. THE TRAINING LOOP ---
# ✅ FIX 4: 300 EPOCHS — loss had not converged at 150
epochs = 300
epoch_losses = []

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        data.x      = torch.nan_to_num(data.x,      nan=0.0)
        data.y      = torch.nan_to_num(data.y,      nan=0.0)
        data.edge_attr = torch.nan_to_num(data.edge_attr, nan=0.0)

        optimizer.zero_grad()

        # ✅ Pass edge_attr to forward() — CGConv requires it
        prediction = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = loss_function(prediction, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    avg_loss = total_loss / len(train_loader.dataset)
    epoch_losses.append(avg_loss)

    # Feed loss into scheduler every epoch
    scheduler.step(avg_loss)

    if epoch % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{epochs} | Loss (MSE): {avg_loss:.4f} | LR: {current_lr:.6f}")

# --- 5. FINAL EVALUATION ---
model.eval()
test_loss     = 0
absolute_error = 0
all_predictions = []
all_actuals     = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        data.x         = torch.nan_to_num(data.x,         nan=0.0)
        data.y         = torch.nan_to_num(data.y,         nan=0.0)
        data.edge_attr = torch.nan_to_num(data.edge_attr, nan=0.0)

        prediction = model(data.x, data.edge_index, data.edge_attr, data.batch)
        test_loss      += loss_function(prediction, data.y).item() * data.num_graphs
        absolute_error += torch.abs(prediction - data.y).sum().item()

        all_predictions.extend(prediction.cpu().numpy().flatten())
        all_actuals.extend(data.y.cpu().numpy().flatten())

# ✅ FIX 3 (cont): Convert log-space back to real GPa for reporting
all_preds_gpa   = np.expm1(np.array(all_predictions))  # inverse of log1p
all_actuals_gpa = np.expm1(np.array(all_actuals))

real_mae = np.mean(np.abs(all_preds_gpa - all_actuals_gpa))
real_mse = np.mean((all_preds_gpa - all_actuals_gpa) ** 2)
r2       = r2_score(all_actuals_gpa, all_preds_gpa)

print("\n--- FINAL RESULTS (in real GPa) ---")
print(f"Test MAE:  {real_mae:.2f} GPa")
print(f"Test MSE:  {real_mse:.2f}")
print(f"R² Score:  {r2:.4f}")

torch.save(model.state_dict(), "crystal_gnn_model.pth")
print("Saved trained model to 'crystal_gnn_model.pth'")

# --- 6. DIAGNOSTICS DASHBOARD ---
print("Generating Diagnostics Dashboard...")

errors = np.abs(all_actuals_gpa - all_preds_gpa)
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Parity Plot (in real GPa)
axs[0].scatter(all_actuals_gpa, all_preds_gpa, alpha=0.6, color='blue', edgecolors='black')
max_val = max(all_actuals_gpa.max(), all_preds_gpa.max())
min_val = min(all_actuals_gpa.min(), all_preds_gpa.min())
axs[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
axs[0].set_title(f'Parity Plot  |  R²={r2:.3f}')
axs[0].set_xlabel('Actual Bulk Modulus (GPa)')
axs[0].set_ylabel('Predicted Bulk Modulus (GPa)')
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.7)

# Plot 2: Training Loss Curve
axs[1].plot(range(1, epochs + 1), epoch_losses, color='purple', lw=2)
axs[1].set_title('Training Loss Curve')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('MSE (log-space)')
axs[1].grid(True, linestyle='--', alpha=0.7)

# Plot 3: Error Distribution
axs[2].hist(errors, bins=30, color='teal', edgecolor='black')
axs[2].axvline(real_mae, color='red', linestyle='--', lw=2, label=f'MAE = {real_mae:.1f} GPa')
axs[2].set_title('Error Distribution')
axs[2].set_xlabel('Absolute Error (GPa)')
axs[2].set_ylabel('Frequency')
axs[2].legend()
axs[2].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("gnn_diagnostics_dashboard.png", dpi=300)
print("✅ Saved diagnostics to 'gnn_diagnostics_dashboard.png'")