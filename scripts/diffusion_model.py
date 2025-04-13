"""
diffusion_model.py
Defines the DDPM model architecture for simulating tumor microenvironment (TME) dynamics.
"""

import torch
import torch.nn as nn

class SimpleDDPM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleDDPM, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t):
        return self.net(x)
