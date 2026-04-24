import torch
from attention_model import AttentionCNN

model = AttentionCNN()

x = torch.randn(1,3,160,160)

emb, attn = model(x)

print("Embedding shape:", emb.shape)
print("Attention map shape:", attn.shape)