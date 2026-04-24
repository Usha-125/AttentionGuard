import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AttentionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.features = nn.Sequential(*list(backbone.children())[:-2])  
        # output: [B, 512, 5, 5] for 160x160 input

        self.attention = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, 128)

    def forward(self, x):
        feat = self.features(x)

        attn = self.attention(feat)
        weighted = feat * attn

        pooled = self.pool(weighted).view(x.size(0), -1)

        emb = self.fc(pooled)
        emb = F.normalize(emb, p=2, dim=1)

        return emb, attn