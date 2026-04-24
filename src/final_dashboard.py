import matplotlib.pyplot as plt

models = ["FaceNet", "Proposed Model"]
accuracy = [98.0, 89.17]

plt.figure(figsize=(7,5))
bars = plt.bar(models, accuracy)

for b in bars:
    y = b.get_height()
    plt.text(b.get_x()+b.get_width()/2, y+0.5, f"{y:.2f}%", ha="center")

plt.ylim(80,100)
plt.ylabel("Accuracy (%)")
plt.title("Final Model Accuracy Comparison")
plt.tight_layout()
plt.show()