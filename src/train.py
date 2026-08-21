"""
Huấn luyện mô hình theo 2 giai đoạn (vẫn là 1 kiến trúc duy nhất):
  Giai đoạn 1: đóng băng VGG16, chỉ train phần CNN phía sau.
  Giai đoạn 2: mở khóa vài layer cuối VGG16, fine-tune với LR nhỏ.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

from src.config import (
    MODELS_DIR, EPOCHS_PHASE1, EPOCHS_PHASE2, FINE_TUNE_AT,
)
from src.data_preprocessing import make_generators, get_class_weights
from src.model import build_combined_model


def get_callbacks(checkpoint_path):
    return [
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, save_weights_only=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]


def run_training():
    os.makedirs(MODELS_DIR, exist_ok=True)
    train_gen, val_gen, test_gen = make_generators()
    class_weights = get_class_weights(train_gen)

    print("=== Huấn luyện mô hình (1 Giai đoạn duy nhất) ===")
    # Khởi tạo mô hình và mở khóa luôn các layer cuối của VGG16 ngay từ đầu
    model = build_combined_model(freeze_base=False, fine_tune_at=FINE_TUNE_AT)
    model.summary()

    # Gop chung số epoch của 2 phase cũ lại
    total_epochs = EPOCHS_PHASE1 + EPOCHS_PHASE2
    
    history_obj = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=total_epochs,
        class_weight=class_weights,
        callbacks=get_callbacks(os.path.join(MODELS_DIR, "final_best.weights.h5")),
        verbose=1,
    )

    # Lưu mô hình cuối cùng + lịch sử train (để notebook vẽ biểu đồ)
    model.save(os.path.join(MODELS_DIR, "final_model.h5"))
    
    # Ép kiểu float32 (numpy) thành native float để ghi vào JSON
    history_dict = {}
    for key in history_obj.history:
        history_dict[key] = [float(v) for v in history_obj.history[key]]
        
    with open(os.path.join(MODELS_DIR, "training_history.json"), "w") as f:
        json.dump(history_dict, f, cls=NpEncoder)

    print(f"\nĐã lưu mô hình tại: {os.path.join(MODELS_DIR, 'final_model.h5')}")
    return model, history_dict, test_gen


if __name__ == "__main__":
    run_training()
