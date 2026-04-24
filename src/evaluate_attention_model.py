import torch
from torch.utils.data import DataLoader
from pair_dataset import FacePairDataset
from attention_model import AttentionCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = FacePairDataset(
    "data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled",
    pairs_count=100
)

loader = DataLoader(dataset, batch_size=8, shuffle=False)

model = AttentionCNN().to(device)
model.load_state_dict(torch.load("models/attention_model.pth"))
model.eval()

best_acc = 0
best_thr = 0

for threshold in [x/100 for x in range(10, 201, 5)]:

    correct = 0
    total = 0

    with torch.no_grad():
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            e1, _ = model(x1)
            e2, _ = model(x2)

            dist = torch.norm(e1 - e2, dim=1)
            pred = (dist < threshold).float()

            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = 100 * correct / total

    if acc > best_acc:
        best_acc = acc
        best_thr = threshold

print("Best Threshold:", best_thr)
print("Best Accuracy:", round(best_acc,2), "%")