import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, 128)

    def forward(self, x):
        feat = self.features(x)

        attn = self.attention(feat)
        weighted = feat * attn

        pooled = self.pool(weighted).view(x.size(0), -1)
        emb = self.fc(pooled)

        emb = F.normalize(emb, p=2, dim=1)

        return emb, attn