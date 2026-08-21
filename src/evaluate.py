"""
Đánh giá mô hình trên tập test:
  - Confusion matrix
  - Precision / Recall / F1-score / classification report
  - ROC-AUC curve
  - Biểu đồ loss & accuracy theo epoch (từ training_history.json)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
    ConfusionMatrixDisplay,
)

from src.config import OUTPUTS_DIR, CLASS_NAMES


def plot_training_history(history, save_path=None):
    """Vẽ biểu đồ loss/accuracy theo epoch. history có thể là dict hoặc keras History."""
    if hasattr(history, "history"):
        history = history.history

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["loss"], label="train_loss")
    axes[0].plot(history["val_loss"], label="val_loss")
    axes[0].set_title("Loss theo epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["accuracy"], label="train_accuracy")
    axes[1].plot(history["val_accuracy"], label="val_accuracy")
    axes[1].set_title("Accuracy theo epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def evaluate_on_test(model, test_gen, save_dir=OUTPUTS_DIR):
    """Đánh giá đầy đủ trên tập test, lưu các biểu đồ vào save_dir."""
    os.makedirs(save_dir, exist_ok=True)

    y_true = test_gen.classes
    y_prob = model.predict(test_gen).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    # --- Classification report (precision, recall, F1) ---
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    print(report)
    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.show()

    # --- ROC-AUC ---
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=150)
    plt.show()

    print(f"AUC: {auc_score:.4f}")
    return {"y_true": y_true, "y_prob": y_prob, "y_pred": y_pred, "auc": auc_score}


def load_history_json(path):
    with open(path, "r") as f:
        return json.load(f)
