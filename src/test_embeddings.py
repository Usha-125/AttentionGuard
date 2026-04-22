from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

mtcnn = MTCNN(image_size=160, margin=20, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

img1_path = Path("data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg")
img2_path = Path("data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled/Aaron_Peirsol/Aaron_Peirsol_0004.jpg")

img1 = Image.open(img1_path)
img2 = Image.open(img2_path)

face1 = mtcnn(img1)
face2 = mtcnn(img2)

face1 = face1.unsqueeze(0).to(device)
face2 = face2.unsqueeze(0).to(device)

emb1 = model(face1)
emb2 = model(face2)

dist = torch.norm(emb1 - emb2).item()

print("Embedding 1 shape:", emb1.shape)
print("Embedding 2 shape:", emb2.shape)
print("Distance:", dist)