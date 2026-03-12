import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Example true labels and predictions
y_true = [0,1,0,1,0,1,1,0]
y_pred = [0,1,0,0,0,1,1,0]

labels = ["fire","no_fire"]

cm = confusion_matrix(y_true, y_pred)

plt.imshow(cm, cmap="Blues")
plt.colorbar()

plt.xticks(range(len(labels)), labels)
plt.yticks(range(len(labels)), labels)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

plt.savefig("confusion_matrix.png")
print("Confusion matrix saved as confusion_matrix.png")
plt.show()