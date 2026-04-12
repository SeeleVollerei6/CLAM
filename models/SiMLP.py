import torch
import torch.nn as nn

class SiMLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=[512, 256], num_classes=2, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, h):
        if h.dim() == 2:
            h = h.unsqueeze(0)
        h = h.squeeze(0)
        bag_rep = torch.mean(h, dim=0)
        logits = self.mlp(bag_rep.unsqueeze(0))
        return logits