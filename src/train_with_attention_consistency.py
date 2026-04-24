import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from pair_dataset import FacePairDataset
from attention_model import AttentionCNN
import matplotlib.pyplot as plt
import os

# =========================
# Device
# =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================
# Hyperparameters
# =========================
pairs_count = 700
batch_size = 16
lr = 0.0003
epochs = 12
lambda_attn = 0.1
margin = 1.0
val_split = 0.2

# =========================
# Dataset
# =========================
dataset = FacePairDataset(
    "data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled",
    pairs_count=pairs_count
)

val_size = int(len(dataset) * val_split)
train_size = len(dataset) - val_size

train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# =========================
# Model
# =========================
model = AttentionCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# =========================
# Loss Function
# =========================
def contrastive_loss(e1, e2, label):
    label = label.float()

    # Normalize embeddings (IMPORTANT)
    e1 = F.normalize(e1, dim=1)
    e2 = F.normalize(e2, dim=1)

    dist = torch.norm(e1 - e2, dim=1)

    same = label * dist.pow(2)
    diff = (1 - label) * torch.clamp(margin - dist, min=0).pow(2)

    return (same + diff).mean()

# =========================
# Accuracy Metric
# =========================
def compute_accuracy(e1, e2, label, threshold=0.5):
    e1 = F.normalize(e1, dim=1)
    e2 = F.normalize(e2, dim=1)

    dist = torch.norm(e1 - e2, dim=1)
    preds = (dist < threshold).float()

    correct = (preds == label).sum().item()
    return correct / len(label)

# =========================
# Training Loop
# =========================
train_history = []
val_history = []
best_val_loss = float('inf')

os.makedirs("models", exist_ok=True)

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x1, x2, y in train_loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        # Adversarial noise
        noise = 0.02 * torch.randn_like(x1)
        x1_adv = torch.clamp(x1 + noise, 0, 1)

        # Forward
        e1, a1 = model(x1)
        e2, _ = model(x2)
        _, a1_adv = model(x1_adv)

        # Loss
        loss_main = contrastive_loss(e1, e2, y)
        loss_attn = F.mse_loss(a1, a1_adv)

        loss = loss_main + lambda_attn * loss_attn

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_history.append(avg_train_loss)

    # =========================
    # Validation
    # =========================
    model.eval()
    val_loss = 0
    val_acc = 0

    with torch.no_grad():
        for x1, x2, y in val_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            e1, _ = model(x1)
            e2, _ = model(x2)

            loss = contrastive_loss(e1, e2, y)
            acc = compute_accuracy(e1, e2, y)

            val_loss += loss.item()
            val_acc += acc

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)

    val_history.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")
    print("-" * 40)

    # =========================
    # Save Best Model
    # =========================
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "models/best_attention_model.pth")
        print("✅ Best model saved!")

# =========================
# Save Final Model
# =========================
torch.save(model.state_dict(), "models/final_attention_model.pth")
print("Final model saved.")

# =========================
# Plot
# =========================
plt.figure()
plt.plot(range(1, epochs+1), train_history, label="Train Loss", marker='o')
plt.plot(range(1, epochs+1), val_history, label="Val Loss", marker='x')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()