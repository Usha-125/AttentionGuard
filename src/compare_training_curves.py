import matplotlib.pyplot as plt

baseline = [7.7315, 6.6970, 6.4971, 6.5640, 6.5564]
proposed = [7.6424, 6.2672, 6.0571, 5.6912, 6.3315]

epochs = [1,2,3,4,5]

plt.figure(figsize=(8,5))

plt.plot(epochs, baseline, marker='o', label="Baseline Attention CNN")
plt.plot(epochs, proposed, marker='o', label="With Attention Consistency")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()