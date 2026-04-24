from pathlib import Path
import random
from PIL import Image
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN
import torch

class FacePairDataset(Dataset):
    def __init__(self, root_dir, pairs_count=500):
        self.root = Path(root_dir)
        self.mtcnn = MTCNN(image_size=160, margin=20)

        persons = [p for p in self.root.iterdir() if p.is_dir()]
        self.usable = []

        for person in persons:
            imgs = list(person.glob("*.jpg"))
            if len(imgs) >= 2:
                self.usable.append(imgs)

        self.samples = []

        # Genuine
        for imgs in self.usable[:pairs_count]:
            a, b = random.sample(imgs, 2)
            self.samples.append((a, b, 1))

        # Impostor
        for _ in range(pairs_count):
            p1, p2 = random.sample(self.usable, 2)
            a = random.choice(p1)
            b = random.choice(p2)
            self.samples.append((a, b, 0))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p1, p2, label = self.samples[idx]

        img1 = Image.open(p1).convert("RGB")
        img2 = Image.open(p2).convert("RGB")

        face1 = self.mtcnn(img1)
        face2 = self.mtcnn(img2)

        return face1, face2, torch.tensor(label, dtype=torch.float32)