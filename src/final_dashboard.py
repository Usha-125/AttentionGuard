import matplotlib.pyplot as plt

models = ["FaceNet", "Attention CNN", "Proposed"]
accuracy = [98.0, 71.5, 64.0]
stability = [0.0, 5.8128e-05, 5.6341e-05]

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
bars = plt.bar(models, accuracy)
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0,100)

for b in bars:
    y = b.get_height()
    plt.text(b.get_x()+b.get_width()/2, y+1, f"{y:.1f}", ha="center")

plt.subplot(1,2,2)
bars2 = plt.bar(models, stability)
plt.title("Attention Drift (Lower Better)")
plt.ylabel("MSE")

for b in bars2:
    y = b.get_height()
    plt.text(b.get_x()+b.get_width()/2, y, f"{y:.1e}", ha="center")

plt.tight_layout()
plt.show()