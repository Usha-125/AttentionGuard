import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(image_size=160, margin=20, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

img1_path = "data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg"
img2_path = "data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled/Aaron_Peirsol/Aaron_Peirsol_0004.jpg"

def load_face(path):
    img = Image.open(path).convert("RGB")
    face = mtcnn(img)
    return face.unsqueeze(0).to(device)

face1 = load_face(img1_path)
face2 = load_face(img2_path)

# Clean embeddings
face1.requires_grad = True

emb1 = model(face1)
emb2 = model(face2)

loss = torch.norm(emb1 - emb2)
model.zero_grad()
loss.backward()

epsilon = 0.02
adv_face1 = face1 + epsilon * face1.grad.sign()
adv_face1 = torch.clamp(adv_face1, 0, 1)

# Compare distances
clean_dist = torch.norm(model(face1) - emb2).item()
adv_dist = torch.norm(model(adv_face1) - emb2).item()

print("Clean Distance:", clean_dist)
print("Adversarial Distance:", adv_dist)

# Show attacked image
img = adv_face1.squeeze().permute(1,2,0).detach().cpu().numpy()

plt.imshow(img)
plt.title("Adversarial Face")
plt.axis("off")
plt.show()
import matplotlib.pyplot as plt

labels = ["Clean", "Adversarial"]
values = [clean_dist, adv_dist]

plt.figure(figsize=(7,5))
bars = plt.bar(labels, values)

for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, y + 0.02, round(y,3),
             ha='center')

plt.ylabel("Embedding Distance")
plt.title("Effect of FGSM Attack on Genuine Pair")
plt.tight_layout()
plt.show()