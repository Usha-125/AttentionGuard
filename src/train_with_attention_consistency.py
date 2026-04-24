import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pair_dataset import FacePairDataset
from attention_model import AttentionCNN
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = FacePairDataset(
    "data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled",
    pairs_count=100
)

loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = AttentionCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

margin = 1.0
lambda_attn = 0.5
epochs = 5

def contrastive_loss(e1, e2, label):
    dist = torch.norm(e1 - e2, dim=1)

    same = label * dist.pow(2)
    diff = (1 - label) * torch.clamp(margin - dist, min=0).pow(2)

    return (same + diff).mean()

history = []

for epoch in range(epochs):
    total_loss = 0

    for x1, x2, y in loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        # Create adversarial-like noisy image
        noise = 0.02 * torch.randn_like(x1)
        x1_adv = torch.clamp(x1 + noise, 0, 1)

        e1, att1 = model(x1)
        e2, att2 = model(x2)
        e1_adv, att1_adv = model(x1_adv)

        loss_main = contrastive_loss(e1, e2, y)

        loss_attn = F.mse_loss(att1, att1_adv)

        loss = loss_main + lambda_attn * loss_attn

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    history.append(total_loss)
    print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "models/attention_consistency_model.pth")
print("Model saved.")

plt.plot(range(1, epochs+1), history, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training with Attention Consistency Loss")
plt.grid(True)
plt.tight_layout()
plt.show()