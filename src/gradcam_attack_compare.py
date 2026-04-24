import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(image_size=160, margin=20, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

img_path = "data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg"

img = Image.open(img_path).convert("RGB")
face = mtcnn(img).unsqueeze(0).to(device)
face.requires_grad = True

# FGSM attack
emb = model(face)
loss = emb.norm()
model.zero_grad()
loss.backward()

epsilon = 0.02
adv_face = face + epsilon * face.grad.sign()
adv_face = torch.clamp(adv_face, 0, 1)

target_layers = [model.block8.branch1[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

clean_cam = cam(input_tensor=face)[0]
adv_cam = cam(input_tensor=adv_face)[0]

clean_img = face.squeeze().permute(1,2,0).detach().cpu().numpy()
adv_img = adv_face.squeeze().permute(1,2,0).detach().cpu().numpy()

clean_vis = show_cam_on_image(np.clip(clean_img,0,1), clean_cam, use_rgb=True)
adv_vis = show_cam_on_image(np.clip(adv_img,0,1), adv_cam, use_rgb=True)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(clean_vis)
plt.title("Clean Heatmap")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(adv_vis)
plt.title("Adversarial Heatmap")
plt.axis("off")

plt.tight_layout()
plt.show()