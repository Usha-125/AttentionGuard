import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from pair_dataset import FacePairDataset
from attention_model import AttentionCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# -----------------------------
# DATASET
# -----------------------------
dataset = FacePairDataset(
    "data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled",
    pairs_count=120
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,      # Windows safe
    pin_memory=True
)

# -----------------------------
# MODEL
# -----------------------------
model = AttentionCNN().to(device)
model.load_state_dict(torch.load("models/best_attention_model.pth"))
model.eval()

# -----------------------------
# CACHE DISTANCES
# -----------------------------
all_distances = []
all_labels = []

with torch.no_grad():
    for x1, x2, y in loader:

        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        e1, _ = model(x1)
        e2, _ = model(x2)

        dist = torch.norm(e1 - e2, dim=1)

        all_distances.extend(dist.cpu().numpy())
        all_labels.extend(y.numpy())

all_distances = np.array(all_distances)
all_labels = np.array(all_labels)

print("Pairs evaluated:", len(all_labels))

# -----------------------------
# THRESHOLD SEARCH
# -----------------------------
best_acc = 0
best_thr = 0
best_pred = None

for threshold in np.arange(0.10, 1.01, 0.02):
    pred = (all_distances < threshold).astype(int)
    acc = accuracy_score(all_labels, pred)

    if acc > best_acc:
        best_acc = acc
        best_thr = threshold
        best_pred = pred

# -----------------------------
# FINAL METRICS
# -----------------------------
precision = precision_score(all_labels, best_pred)
recall = recall_score(all_labels, best_pred)
f1 = f1_score(all_labels, best_pred)
cm = confusion_matrix(all_labels, best_pred)

print("\nBest Threshold:", round(best_thr, 2))
print("Accuracy:", round(best_acc * 100, 2), "%")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

# -----------------------------
# PLOT
# -----------------------------
plt.figure(figsize=(5,4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks([0,1], ["Different", "Same"])
plt.yticks([0,1], ["Different", "Same"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha='center', va='center')

plt.tight_layout()
plt.show()