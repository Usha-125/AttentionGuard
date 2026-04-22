from pathlib import Path
import random
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(image_size=160, margin=20, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

THRESHOLD = 1.1
DATASET = Path("data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled")

def emb(path):
    img = Image.open(path)
    face = mtcnn(img)
    face = face.unsqueeze(0).to(device)
    return model(face)

def predict(p1, p2):
    e1 = emb(p1)
    e2 = emb(p2)
    dist = torch.norm(e1 - e2).item()
    return 1 if dist < THRESHOLD else 0

persons = [p for p in DATASET.iterdir() if p.is_dir()]
usable = []

for person in persons:
    imgs = list(person.glob("*.jpg"))
    if len(imgs) >= 2:
        usable.append(imgs)

pairs = []

# 100 genuine
for imgs in usable[:100]:
    a, b = random.sample(imgs, 2)
    pairs.append((a, b, 1))

# 100 impostor
for _ in range(100):
    p1, p2 = random.sample(usable, 2)
    pairs.append((random.choice(p1), random.choice(p2), 0))

correct = 0

for a, b, label in pairs:
    pred = predict(a, b)
    if pred == label:
        correct += 1

acc = correct / len(pairs) * 100
print("Total Pairs:", len(pairs))
print("Correct:", correct)
print("Accuracy:", round(acc, 2), "%")