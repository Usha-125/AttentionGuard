import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pair_dataset import FacePairDataset
from attention_model import AttentionCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = FacePairDataset(
    "data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled",
    pairs_count=100
)

loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = AttentionCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

margin = 1.0

def contrastive_loss(e1, e2, label):
    dist = torch.norm(e1 - e2, dim=1)

    loss_same = label * dist.pow(2)
    loss_diff = (1 - label) * torch.clamp(margin - dist, min=0).pow(2)

    return (loss_same + loss_diff).mean()

epochs = 5

for epoch in range(epochs):
    total_loss = 0

    for x1, x2, y in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)

        e1, a1 = model(x1)
        e2, a2 = model(x2)

        loss = contrastive_loss(e1, e2, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "models/attention_model.pth")
print("Model saved.")