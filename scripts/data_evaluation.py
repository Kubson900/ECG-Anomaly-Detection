import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn.metrics import f1_score


def plot_confusion_matrix(labels, outputs, class_names, file_path=None):
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
    if file_path:
        plt.savefig(file_path)
    plt.show()


def plot_optimal_thresholds(
    y_test: np.ndarray, y_pred_proba: np.ndarray, file_path=None
) -> list[float]:
    thresholds = []
    f1_scores = []
    optimal_thresholds = []
    number_of_classes = 5
    class_names = ["CD", "HYP", "MI", "NORM", "STTC"]

    for class_index in range(number_of_classes):
        class_thresholds = []
        class_f1_scores = []

        for threshold in np.arange(0, 1.0, 0.05):
            y_pred = (y_pred_proba[:, class_index] > threshold) * 1
            f1 = f1_score(y_test[:, class_index], y_pred, average="weighted")

            class_thresholds.append(threshold)
            class_f1_scores.append(f1)

        optimal_threshold = np.round(
            class_thresholds[np.argmax(class_f1_scores)], decimals=2
        )

        thresholds.append(class_thresholds)
        f1_scores.append(class_f1_scores)
        optimal_thresholds.append(optimal_threshold)

    f, axes = plt.subplots(5, 1, figsize=(15, 15))
    axes = axes.ravel()
    for i in range(number_of_classes):
        axes[i].plot(thresholds[i], f1_scores[i])
        axes[i].vlines(optimal_thresholds[i], 0, 1, linestyles="--", color="black")
        axes[i].set_title(class_names[i], fontsize=16)
        axes[i].set_xlabel("Threshold")
        axes[i].set_ylabel("F1 score")
        axes[i].set_xlim(([0, 1]))
    plt.subplots_adjust(wspace=0.5, hspace=1)

    if file_path:
        plt.savefig(file_path, bbox_inches="tight")

    plt.show()
    return optimal_thresholds
