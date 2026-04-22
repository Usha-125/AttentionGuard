from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(image_size=160, margin=20, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(path):
    img = Image.open(path)
    face = mtcnn(img)
    face = face.unsqueeze(0).to(device)
    emb = model(face)
    return emb

def distance(p1, p2):
    e1 = get_embedding(p1)
    e2 = get_embedding(p2)
    return torch.norm(e1 - e2).item()

same1 = "data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg"
same2 = "data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled/Aaron_Peirsol/Aaron_Peirsol_0004.jpg"

diff1 = same1
diff2 = "data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled/Abba_Eban/Abba_Eban_0001.jpg"

print("Same Person Distance:", distance(same1, same2))
print("Different Person Distance:", distance(diff1, diff2))