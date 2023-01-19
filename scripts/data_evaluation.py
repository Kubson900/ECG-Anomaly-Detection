from typing import Union

import tensorflow as tf
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import rcParams
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import numpy as np
from sklearn.metrics import f1_score
import os
from tensorflow._api.v2.v2.keras import Model

# Add every font at the specified location
font_dir = ["fonts"]
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)

# Set font family globally
rcParams["font.family"] = "umr10"
rcParams["font.size"] = 16


def plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    directory_name: str,
    file_name: str = "confusion_matrix",
    save_plot: bool = False,
):
    fig = plt.figure(figsize=(20, 10), dpi=400)
    spec = matplotlib.gridspec.GridSpec(ncols=6, nrows=2)  # 6 columns evenly divides both 2 & 3

    ax1 = fig.add_subplot(spec[0, 0:2])  # row 0 with axes spanning 2 cols on evens
    ax2 = fig.add_subplot(spec[0, 2:4])
    ax3 = fig.add_subplot(spec[0, 4:])
    ax4 = fig.add_subplot(spec[1, 1:3])  # row 0 with axes spanning 2 cols on odds
    ax5 = fig.add_subplot(spec[1, 3:5])

    axes = [ax1, ax2, ax3, ax4, ax5]

    for i in range(5):
        disp = ConfusionMatrixDisplay(
            confusion_matrix(y_test[:, i], y_pred[:, i])
        )
        disp.plot(ax=axes[i], values_format=".4g", cmap="viridis")
        disp.ax_.set_title(f"Class {class_names[i]}")
        disp.im_.colorbar.remove()

    plt.subplots_adjust(wspace=8, hspace=1)
    fig.colorbar(disp.im_, ax=axes)

    if save_plot:
        output_directory = f"saved_images/{directory_name}/"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory, exist_ok=True)
        file_path = os.path.join(output_directory, file_name)
        plt.savefig(file_path, bbox_inches="tight", dpi=400)
    plt.show()


def plot_optimal_thresholds(
    y_test: np.ndarray,
    y_pred_proba: np.ndarray,
    directory_name: str,
    file_name: str = "optimal_thresholds",
    save_plot: bool = False,
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
        axes[i].set_title(class_names[i], fontsize=24)
        axes[i].set_xlabel("Threshold")
        axes[i].set_ylabel("F1 score")
        axes[i].set_xlim(([0, 1]))
    plt.subplots_adjust(wspace=0.5, hspace=1)

    if save_plot:
        output_directory = f"optimal_thresholds_visualizations/{directory_name}/"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory, exist_ok=True)
        file_path = os.path.join(output_directory, file_name)
        plt.savefig(file_path, bbox_inches="tight", dpi=400)

    plt.show()
    return optimal_thresholds


def generate_model_evaluation(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    directory_name: str,
    file_name: str = "model_evaluation",
    save_data: bool = False,
) -> pd.DataFrame:
    binary_crossentropy_loss = tf.keras.losses.BinaryCrossentropy()
    loss = binary_crossentropy_loss(y_test, y_pred_proba)

    binary_accuracy = tf.keras.metrics.BinaryAccuracy()
    _ = binary_accuracy.update_state(y_test, y_pred)

    recall = tf.keras.metrics.Recall()
    _ = recall.update_state(y_test, y_pred)

    precision = tf.keras.metrics.Precision()
    _ = precision.update_state(y_test, y_pred)

    auc = tf.keras.metrics.AUC(multi_label=True)
    _ = auc.update_state(y_test, y_pred)

    model_evaluation = np.array(
        [
            loss.numpy(),
            binary_accuracy.result().numpy(),
            recall.result().numpy(),
            precision.result().numpy(),
            auc.result().numpy(),
        ]
    )
    model_evaluation = np.round(model_evaluation, 3)
    model_evaluation_df = pd.DataFrame(
        data=model_evaluation,
        index=["loss", "binary_accuracy", "recall", "precision", "auc"],
    ).transpose()

    if save_data:
        output_directory = f"saved_data/{directory_name}/"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory, exist_ok=True)
        model_evaluation_df.to_csv(os.path.join(output_directory, file_name))

    return model_evaluation_df


def generate_classification_report(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    directory_name: str,
    file_name: str = "classification_report",
    save_data: bool = False,
) -> pd.DataFrame:
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True,
    )
    report_df = pd.DataFrame(report).transpose().round(decimals=3)

    if save_data:
        report_df.to_csv(f"saved_data/{directory_name}/{file_name}")

    return report_df


def generate_sample_patients_predictions(
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: Model,
    threshold: Union[float, np.array],
    class_names: list[str],
    number_of_patients: int,
    directory_name: str,
    file_name: str = "sample_patients_predictions",
    save_data: bool = False,
) -> pd.DataFrame:

    sample_patients_predictions_df = pd.DataFrame()

    for index in range(number_of_patients):
        patient_ecg = np.expand_dims(X_test[index], axis=0)
        patient_ecg_prob = (model.predict(patient_ecg) > threshold) * 1
        sample_patients_predictions_df = pd.concat(
            [
                sample_patients_predictions_df,
                pd.DataFrame(
                    data={
                        "Predicted": np.squeeze(patient_ecg_prob),
                        "True": y_test[index],
                        "---------": np.array(["-" for x in range(5)]),
                    },
                    index=class_names,
                ).transpose(),
            ]
        )

    if save_data:
        sample_patients_predictions_df.to_csv(
            f"saved_data/{directory_name}/{file_name}"
        )

    return sample_patients_predictions_df
