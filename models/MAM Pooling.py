import torch
import torch.nn as nn

class MAM_Classifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, n_classes=2):
        super(MAM_Classifier, self).__init__()
        
        self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        
        self.cls_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, h):
        # h shape: [N, 1024]
        h = self.fc1(h) # [N, 512]
        
        #y = (max(x) + mean(x) + min(x)) / 3
        max_p = torch.max(h, dim=0)[0]
        mean_p = torch.mean(h, dim=0)
        min_p = torch.min(h, dim=0)[0]
        
        z = (max_p + mean_p + min_p) / 3 # [512]
        
        logits = self.cls_head(z.unsqueeze(0)) # [1, n_classes]
        return logits