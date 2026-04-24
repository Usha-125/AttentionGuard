import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(image_size=160, margin=20, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Choose image
img_path = "data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg"

img = Image.open(img_path).convert("RGB")
face = mtcnn(img)
input_tensor = face.unsqueeze(0).to(device)

# Last convolution-like target layer
target_layers = [model.block8.branch1[-1]]

cam = GradCAM(model=model, target_layers=target_layers)

grayscale_cam = cam(input_tensor=input_tensor)[0]

rgb_img = face.permute(1,2,0).cpu().numpy()
rgb_img = np.clip(rgb_img, 0, 1)

visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(rgb_img)
plt.title("Original Face")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(visualization)
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.tight_layout()
plt.show()