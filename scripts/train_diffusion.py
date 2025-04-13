"""
train_diffusion.py
Training loop for the DDPM model on single-cell RNA-seq data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from diffusion_model import SimpleDDPM
from sklearn.preprocessing import StandardScaler
import scanpy as sc

def load_data(path):
    adata = sc.read_10x_mtx(path, var_names='gene_symbols', cache=True)
    adata = adata[adata.obs.n_counts < 2500, :]
    return adata

def train(model, data, epochs=100, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        t = torch.randint(0, 1000, (data.size(0),)).long()
        output = model(data, t)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    adata = load_data("data/pbmc_3k/")
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    model = SimpleDDPM(input_dim=X_tensor.shape[1], hidden_dim=256)
    train(model, X_tensor)
