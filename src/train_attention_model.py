import torch
from torch.utils.data import DataLoader
from pair_dataset import FacePairDataset
from attention_model import AttentionCNN
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

dataset = FacePairDataset(
    "data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled",
    pairs_count=700
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

model = AttentionCNN().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0003
)

margin = 1.0
epochs = 12

def contrastive_loss(e1, e2, label):
    dist = torch.norm(e1 - e2, dim=1)

    same = label * dist.pow(2)
    diff = (1 - label) * torch.clamp(margin - dist, min=0).pow(2)

    return (same + diff).mean()

history = []

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x1, x2, y in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)

        e1, _ = model(x1)
        e2, _ = model(x2)

        loss = contrastive_loss(e1, e2, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    history.append(total_loss)

    print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "models/attention_model.pth")
print("ResNet18 Attention model saved.")

plt.figure(figsize=(8,5))
plt.plot(range(1, epochs+1), history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet18 Attention Training Curve")
plt.grid(True)
plt.tight_layout()
plt.show()