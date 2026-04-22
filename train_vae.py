import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

# 1. Define the Neural Network Architecture
class MaterialVAE(nn.Module):
    def __init__(self, input_dim=88, latent_dim=4):
        super(MaterialVAE, self).__init__()
        # ENCODER: Compresses 88 elements down to a 4-dimensional pattern
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2_mu = nn.Linear(64, latent_dim)
        self.fc2_logvar = nn.Linear(64, latent_dim)
        
        # DECODER: Expands the 4-dimensional pattern back into 88 elements
        self.fc3 = nn.Linear(latent_dim, 64)
        self.fc4 = nn.Linear(64, input_dim)
        
    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc2_mu(h1), self.fc2_logvar(h1)
        
    def reparameterize(self, mu, logvar):
        """This 'trick' allows the AI to generate variations of materials."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        # Sigmoid ensures the output fractions are between 0 and 1
        return torch.sigmoid(self.fc4(h3))
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# 2. Define the training process
class VAETrainer:
    def __init__(self, data_file="engineered_materials.csv"):
        self.data_file = data_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def loss_function(self, recon_x, x, mu, logvar):
        """Measures how well the AI reconstructs the material + how well it learned the patterns."""
        BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
        # Kullback-Leibler divergence (forces the latent space to be continuous)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def train_and_save(self, epochs=50, batch_size=128, save_path="material_vae.pth"):
        print(f"1. Loading chemical data for Neural Network from {self.data_file}...")
        df = pd.read_csv(self.data_file)
        
        # We only want the 88 elemental features to train the generator
        X = df.drop(columns=['Formula', 'Density', 'Bulk_Modulus', 'Shear_Modulus']).values
        
        # Convert to PyTorch Tensors
        tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(tensor_x)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        input_dimensions = X.shape[1]
        self.model = MaterialVAE(input_dim=input_dimensions).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        print(f"2. Training VAE on {self.device} for {epochs} epochs...")
        self.model.train()
        
        for epoch in range(epochs):
            train_loss = 0
            for batch in dataloader:
                batch_data = batch[0]
                optimizer.zero_grad()
                
                # Forward pass
                reconstructed_batch, mu, logvar = self.model(batch_data)
                
                # Calculate loss and update weights
                loss = self.loss_function(reconstructed_batch, batch_data, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch + 1}/{epochs} | Average Loss: {train_loss / len(dataloader.dataset):.4f}")

        print("3. Training complete! Saving generator weights for the final backend...")
        # We save the state dictionary (the AI's learned "brain") permanently
        torch.save(self.model.state_dict(), save_path)
        print(f"Success! Generative model securely saved as '{save_path}'.")

if __name__ == "__main__":
    trainer = VAETrainer()
    trainer.train_and_save()