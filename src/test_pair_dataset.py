from pair_dataset import FacePairDataset

dataset = FacePairDataset(
    "data/raw/lfw/lfw-deepfunneled/lfw-deepfunneled",
    pairs_count=10
)

print("Dataset size:", len(dataset))

x1, x2, y = dataset[0]

print("Face1 shape:", x1.shape)
print("Face2 shape:", x2.shape)
print("Label:", y.item())