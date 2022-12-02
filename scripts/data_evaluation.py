import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


def compute_accuracy(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    num_correct_recordings = 0
    for i in range(num_recordings):
        if np.all(labels[i, :] == outputs[i, :]):
            num_correct_recordings += 1

    return float(num_correct_recordings) / float(num_recordings)


def plot_confusion_matrix(labels, outputs, class_names):
    f, axes = plt.subplots(1, 5, figsize=(15, 5))
    axes = axes.ravel()
    for i in range(5):
        disp = ConfusionMatrixDisplay(
            confusion_matrix(labels[:, i], outputs[:, i]), display_labels=[0, i]
        )
        disp.plot(ax=axes[i], values_format=".4g", cmap="magma")
        disp.ax_.set_title(f"Class {class_names[i]}")
        disp.im_.colorbar.remove()

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    f.colorbar(disp.im_, ax=axes)
    plt.show()
