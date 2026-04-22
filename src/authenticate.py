from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(image_size=160, margin=20, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

THRESHOLD = 1.1

def get_embedding(path):
    img = Image.open(path)
    face = mtcnn(img)
    face = face.unsqueeze(0).to(device)
    return model(face)

def authenticate(img1, img2):
    e1 = get_embedding(img1)
    e2 = get_embedding(img2)

    dist = torch.norm(e1 - e2).item()

    if dist < THRESHOLD:
        result = "MATCH"
    else:
        result = "NOT MATCH"

    print("Distance:", dist)
    print("Prediction:", result)

img1 = "data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg"
img2 = "data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled/Aaron_Peirsol/Aaron_Peirsol_0004.jpg"

authenticate(img1, img2)