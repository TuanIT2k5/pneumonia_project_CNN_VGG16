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
    ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
)

from src.config import OUTPUTS_DIR, CLASS_NAMES


def plot_training_history(history, save_path=None, show=True):
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
    if show:
        try:
            plt.show()
        except:
            pass
    plt.close()


def evaluate_on_test(model, test_gen, save_dir=OUTPUTS_DIR, threshold=0.5, run_name="", show=True):
    """Đánh giá đầy đủ trên tập test, lưu các biểu đồ vào save_dir."""
    os.makedirs(save_dir, exist_ok=True)
    prefix = f"{run_name}_" if run_name else ""

    y_true = test_gen.classes
    y_prob = model.predict(test_gen).ravel()

    # 1. Tính toán ROC và ngưỡng tối ưu (Youden's J)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    J = tpr - fpr
    optimal_idx = np.argmax(J)
    optimal_threshold = roc_thresholds[optimal_idx]

    # Sử dụng ngưỡng cố định là 0.5 nếu không truyền threshold
    used_threshold = threshold if threshold is not None else 0.5

    print(f"Gợi ý ngưỡng tối ưu (Youden's J): {optimal_threshold:.4f} (đang dùng threshold={used_threshold:.4f})")

    y_pred = (y_prob >= used_threshold).astype(int)


    # 4. Lưu lại danh sách ảnh dự đoán sai
    misclassified_idx = np.where(y_true != y_pred)[0]
    misclassified_files = []
    if hasattr(test_gen, "filenames") or hasattr(test_gen, "filepaths"):
        files_attr = test_gen.filenames if hasattr(test_gen, "filenames") else test_gen.filepaths
        misclassified_files = [files_attr[i] for i in misclassified_idx]
        with open(os.path.join(save_dir, f"{prefix}misclassified.txt"), "w") as f:
            for mf in misclassified_files:
                f.write(f"{mf}\n")

    # 2. Lưu lại dữ liệu thô (.npz)
    np.savez(
        os.path.join(save_dir, f"{prefix}raw_predictions.npz"), 
        y_true=y_true, y_prob=y_prob, fpr=fpr, tpr=tpr
    )

    # --- Classification report (precision, recall, F1) ---
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    print(report)
    with open(os.path.join(save_dir, f"{prefix}classification_report.txt"), "w") as f:
        f.write(report)
        f.write(f"\nThreshold used: {used_threshold:.4f}\nAUC: {auc_score:.4f}\nOptimal Threshold (Youden's J): {optimal_threshold:.4f}\n")

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix (Thresh={used_threshold:.4f})")
    plt.savefig(os.path.join(save_dir, f"{prefix}confusion_matrix.png"), dpi=150)
    if show:
        try: plt.show()
        except: pass
    plt.close()

    # --- ROC-AUC ---
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', label=f'Best Threshold = {optimal_threshold:.2f}')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{prefix}roc_curve.png"), dpi=150)
    if show:
        try: plt.show()
        except: pass
    plt.close()
    
    # 6. Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, label=f"Avg Precision = {avg_precision:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{prefix}pr_curve.png"), dpi=150)
    if show:
        try: plt.show()
        except: pass
    plt.close()

    print(f"AUC: {auc_score:.4f}")
    return {
        "y_true": y_true, "y_prob": y_prob, "y_pred": y_pred, 
        "auc": auc_score, "optimal_threshold": optimal_threshold,
        "misclassified_files": misclassified_files
    }


def load_history_json(path):
    with open(path, "r") as f:
        return json.load(f)
