import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from attention_model import AttentionCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(image_size=160, margin=20, device=device)
model = AttentionCNN().to(device).eval()

img_path = "data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg"

img = Image.open(img_path).convert("RGB")
face = mtcnn(img)

x = face.unsqueeze(0).to(device)

with torch.no_grad():
    emb, attn = model(x)

img_np = face.permute(1,2,0).cpu().numpy()
attn_map = attn.squeeze().cpu().numpy()

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(np.clip(img_np,0,1))
plt.title("Input Face")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(attn_map, cmap="jet")
plt.title("Attention Map")
plt.axis("off")

plt.tight_layout()
plt.show()