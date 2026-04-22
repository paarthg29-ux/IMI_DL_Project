import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import json

class MaterialVAE(nn.Module):
    def __init__(self, input_dim=88, latent_dim=4):
        super(MaterialVAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2_mu = nn.Linear(64, latent_dim)
        self.fc2_logvar = nn.Linear(64, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 64)
        self.fc4 = nn.Linear(64, input_dim)
        
    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc2_mu(h1), self.fc2_logvar(h1)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class VAETrainer:
    def __init__(self, data_file="engineered_materials.csv"):
        self.data_file = data_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD, BCE, KLD

    def train_and_save(self, epochs=50, batch_size=128, save_path="material_vae.pth"):
        print(f"1. Loading data from {self.data_file}...")
        df = pd.read_csv(self.data_file)
        X = df.drop(columns=['Formula', 'Density', 'Bulk_Modulus', 'Shear_Modulus']).values
        
        tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(tensor_x)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = MaterialVAE(input_dim=X.shape[1]).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        print(f"2. Training VAE on {self.device}...")
        self.model.train()

        loss_history = []

        for epoch in range(epochs):
            train_loss, train_bce, train_kld = 0, 0, 0
            for batch in dataloader:
                batch_data = batch[0]
                optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(batch_data)
                
                loss, bce, kld = self.loss_function(recon_batch, batch_data, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                train_bce += bce.item()
                train_kld += kld.item()
                optimizer.step()

            avg_loss = train_loss / len(dataloader.dataset)
            # ── UI HOOK RESTORED ──
            loss_history.append({"epoch": epoch + 1, "loss": round(avg_loss, 6)})

            if (epoch + 1) % 10 == 0:
                n_samp = len(dataloader.dataset)
                print(f"   Epoch {epoch + 1}/{epochs} | Total: {avg_loss:.4f} | Recon: {train_bce/n_samp:.4f} | KLD: {train_kld/n_samp:.4f}")

        print("3. Saving generator weights...")
        torch.save(self.model.state_dict(), save_path)

        # ── UI HOOKS RESTORED ──
        print("4. Saving UI Data Files (Latent Space & Loss History)...")
        self.model.eval()
        with torch.no_grad():
            tensor_all = torch.tensor(X, dtype=torch.float32).to(self.device)
            mu_all, _ = self.model.encode(tensor_all)
            latent_coords = mu_all.cpu().numpy()

        latent_df = pd.DataFrame(latent_coords, columns=[f"z{i+1}" for i in range(latent_coords.shape[1])])
        latent_df[['z1', 'z2']].to_csv("vae_latent_space.csv", index=False)
        
        with open("vae_loss_history.json", "w") as f:
            json.dump(loss_history, f)
            
        print("Success! UI hooks saved.")

if __name__ == "__main__":
    trainer = VAETrainer()
    trainer.train_and_save()