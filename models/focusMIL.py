import torch
import torch.nn as nn
import torch.nn.functional as F

class FocusMIL(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=[512, 256], latent_dim=256, num_classes=2, beta=0.001):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, h):
        if h.dim() == 2:
            h = h.unsqueeze(0)
        h = h.squeeze(0)
        encoded = self.encoder(h)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        bag_rep, _ = torch.max(z, dim=0)
        logits = self.classifier(bag_rep.unsqueeze(0))
        return logits, mu, logvar