import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import MTCNN
from attention_model import AttentionCNN
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(image_size=160, margin=20, device=device)

img_path = "data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg"

img = Image.open(img_path).convert("RGB")
face = mtcnn(img).unsqueeze(0).to(device)

noise = 0.02 * torch.randn_like(face)
face_adv = torch.clamp(face + noise, 0, 1)

def get_mse(model_path):
    model = AttentionCNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        _, a1 = model(face)
        _, a2 = model(face_adv)

    return F.mse_loss(a1, a2).item()

baseline_mse = get_mse("models/attention_model.pth")
proposed_mse = get_mse("models/attention_consistency_model.pth")

print("Baseline MSE:", baseline_mse)
print("Proposed MSE:", proposed_mse)

plt.bar(
    ["Baseline", "Proposed"],
    [baseline_mse, proposed_mse]
)

plt.ylabel("Attention Drift MSE")
plt.title("Attention Stability Comparison")
plt.tight_layout()
plt.show()