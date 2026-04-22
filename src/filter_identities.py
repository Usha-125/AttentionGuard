from pathlib import Path

DATASET = Path("data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled")

persons = [p for p in DATASET.iterdir() if p.is_dir()]

usable = []

for person in persons:
    imgs = list(person.glob("*.jpg"))
    if len(imgs) >= 2:
        usable.append((person.name, len(imgs)))

print("Total identities:", len(persons))
print("Usable identities (2+ images):", len(usable))

print("\nSample usable identities:")
for name, count in usable[:10]:
    print(name, ":", count)