from pathlib import Path
import random
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# -----------------------------
# Device Setup
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# -----------------------------
# Models
# -----------------------------
mtcnn = MTCNN(image_size=160, margin=20, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# -----------------------------
# Config
# -----------------------------
THRESHOLD = 1.1
DATASET = Path("data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled")

# -----------------------------
# Functions
# -----------------------------
def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    face = mtcnn(img)

    if face is None:
        return None

    face = face.unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(face)

    return embedding


def predict(img1, img2):
    emb1 = get_embedding(img1)
    emb2 = get_embedding(img2)

    if emb1 is None or emb2 is None:
        return -1

    distance = torch.norm(emb1 - emb2).item()

    if distance < THRESHOLD:
        return 1
    else:
        return 0


# -----------------------------
# Load Dataset
# -----------------------------
persons = [p for p in DATASET.iterdir() if p.is_dir()]

usable = []

for person in persons:
    imgs = list(person.glob("*.jpg"))
    if len(imgs) >= 2:
        usable.append(imgs)

print("Total usable identities:", len(usable))

# -----------------------------
# Generate Pairs
# -----------------------------
pairs = []

# Genuine pairs
for imgs in usable[:100]:
    a, b = random.sample(imgs, 2)
    pairs.append((a, b, 1))

# Impostor pairs
for _ in range(100):
    p1, p2 = random.sample(usable, 2)
    a = random.choice(p1)
    b = random.choice(p2)
    pairs.append((a, b, 0))

# -----------------------------
# Evaluation
# -----------------------------
y_true = []
y_pred = []
correct = 0
skipped = 0

for a, b, label in pairs:
    pred = predict(a, b)

    if pred == -1:
        skipped += 1
        continue

    y_true.append(label)
    y_pred.append(pred)

    if pred == label:
        correct += 1

total = len(y_true)
accuracy = (correct / total) * 100 if total > 0 else 0

print("\nTotal Pairs Generated:", len(pairs))
print("Pairs Evaluated:", total)
print("Pairs Skipped:", skipped)
print("Correct Predictions:", correct)
print("Accuracy:", round(accuracy, 2), "%")

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Different", "Same"]
)

disp.plot(cmap="Blues")
plt.title("Baseline Face Authentication Confusion Matrix")
plt.tight_layout()
plt.show()