import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, out_dim=40):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 360),
            nn.ReLU(True),
            nn.Linear(360, 120),
            nn.ReLU(True),
            nn.Linear(120, out_dim),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, 120),
            nn.ReLU(True),
            nn.Linear(120, 360),
            nn.ReLU(True),
            nn.Linear(360, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
