import matplotlib.pyplot as plt

methods = [
    "FaceNet\nBaseline",
    "Attention\nCNN",
    "Proposed\nModel"
]

accuracy = [98.0, 95.0, 96.5]

plt.figure(figsize=(8,5))

bars = plt.bar(methods, accuracy)

for bar in bars:
    y = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        y + 0.2,
        f"{y}%",
        ha="center"
    )

plt.ylim(90, 100)
plt.ylabel("Accuracy (%)")
plt.title("Model Performance Comparison")
plt.tight_layout()
plt.show()
