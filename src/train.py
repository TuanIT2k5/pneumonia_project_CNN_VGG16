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
    MODELS_DIR, EPOCHS_PHASE1, EPOCHS_PHASE2, FINE_TUNE_AT, USE_OVERSAMPLING,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, REDUCE_LR_MIN_DELTA,
)
from src.data_preprocessing import make_generators, make_oversampled_generator, get_class_weights
from src.model import build_combined_model


def get_callbacks(checkpoint_path):
    return [
        callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_auc", mode="max",
            save_best_only=True, save_weights_only=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_delta=REDUCE_LR_MIN_DELTA,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def run_training():
    os.makedirs(MODELS_DIR, exist_ok=True)

    steps_per_epoch = None  # None → Keras tự tính

    if USE_OVERSAMPLING:
        print("=== Chế độ: OVERSAMPLING NORMAL ===")
        train_gen, val_gen, test_gen, steps_per_epoch = make_oversampled_generator()
        # Với oversampling, không truyền class_weight (việc cân bằng đã nằm trong generator)
        class_weights = None
        # Lấy một train_gen thường để hiển thị class indices
        _tmp_gen, _, _ = make_generators()
        print("Class indices:", _tmp_gen.class_indices)
    else:
        print("=== Chế độ: CLASS WEIGHT BOOST ===")
        train_gen, val_gen, test_gen = make_generators()
        class_weights = get_class_weights(train_gen)

    print("\n" + "="*50)
    print("=== GIAI ĐOẠN 1: Đóng băng VGG16 (Warm-up) ===")
    print("="*50)
    # Khởi tạo mô hình ở chế độ freeze_base=True (chỉ train custom layers)
    model = build_combined_model(freeze_base=True)
    model.summary()

    history_phase1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE1,
        steps_per_epoch=steps_per_epoch,  # None khi không oversample
        class_weight=class_weights,
        callbacks=get_callbacks(os.path.join(MODELS_DIR, "phase1_best.weights.h5")),
        verbose=1,
    )

    print("\n" + "="*50)
    print("=== GIAI ĐOẠN 2: Fine-Tuning VGG16 ===")
    print("="*50)
    
    # Mở khóa VGG16 từ layer FINE_TUNE_AT
    vgg_layer = model.get_layer("vgg16")
    vgg_layer.trainable = True
    for layer in vgg_layer.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    # Compile lại mô hình sau khi unfreeze (bắt buộc)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=[
            "accuracy", 
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    history_phase2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE2,
        steps_per_epoch=steps_per_epoch,
        class_weight=class_weights,
        callbacks=get_callbacks(os.path.join(MODELS_DIR, "final_best.weights.h5")),
        verbose=1,
    )

    # Lưu mô hình cuối cùng + lịch sử train (để notebook vẽ biểu đồ)
    model.save(os.path.join(MODELS_DIR, "final_model.h5"))
    
    # Nối history 2 phase với nhau ép thành native float
    history_dict = {}
    for key in history_phase1.history:
        h1 = [float(v) for v in history_phase1.history[key]]
        h2 = [float(v) for v in history_phase2.history[key]]
        history_dict[key] = h1 + h2
        
    with open(os.path.join(MODELS_DIR, "training_history.json"), "w") as f:
        json.dump(history_dict, f, cls=NpEncoder)

    print(f"\nĐã lưu mô hình tại: {os.path.join(MODELS_DIR, 'final_model.h5')}")
    return model, history_dict, test_gen


if __name__ == "__main__":
    run_training()
