from pathlib import Path
import random

DATASET = Path("data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled")

persons = [p for p in DATASET.iterdir() if p.is_dir()]

# Keep only identities with 2+ images
usable = []
for person in persons:
    imgs = list(person.glob("*.jpg"))
    if len(imgs) >= 2:
        usable.append((person.name, imgs))

genuine_pairs = []
impostor_pairs = []

# Genuine pairs
for name, imgs in usable[:100]:
    pair = random.sample(imgs, 2)
    genuine_pairs.append((pair[0], pair[1], 1))

# Impostor pairs
for _ in range(100):
    p1, p2 = random.sample(usable, 2)
    img1 = random.choice(p1[1])
    img2 = random.choice(p2[1])
    impostor_pairs.append((img1, img2, 0))

print("Genuine pairs:", len(genuine_pairs))
print("Impostor pairs:", len(impostor_pairs))

print("\nSample Genuine Pair:")
print(genuine_pairs[0])

print("\nSample Impostor Pair:")
print(impostor_pairs[0])