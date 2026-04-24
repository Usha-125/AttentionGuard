import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import MTCNN
from attention_model import AttentionCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(image_size=160, margin=20, device=device)

img_path = "data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg"

img = Image.open(img_path).convert("RGB")
face = mtcnn(img).unsqueeze(0).to(device)

# Load trained model
model = AttentionCNN().to(device)
model.load_state_dict(torch.load("models/attention_consistency_model.pth"))
model.eval()

with torch.no_grad():
    _, att_clean = model(face)

noise = 0.02 * torch.randn_like(face)
face_adv = torch.clamp(face + noise, 0, 1)

with torch.no_grad():
    _, att_adv = model(face_adv)

mse = F.mse_loss(att_clean, att_adv).item()

print("Attention Stability MSE:", mse)