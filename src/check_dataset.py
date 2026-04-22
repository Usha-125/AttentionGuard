#How many identity folders exist?
#How many images inside each folder?
from pathlib import Path

DATASET = Path("data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled")

print("Checking LFW dataset...\n")

if not DATASET.exists():
    print("Folder not found.")
else:
    persons = [p for p in DATASET.iterdir() if p.is_dir()]
    print("Total identity folders:", len(persons))

    for person in persons[:10]:
        imgs = list(person.glob("*.jpg"))
        print(f"{person.name}: {len(imgs)} images")

