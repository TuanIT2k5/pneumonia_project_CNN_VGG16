"""
Tiền xử lý dữ liệu: tạo ImageDataGenerator cho train/val/test,
tính class_weight để xử lý mất cân bằng lớp.
"""

import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE, SEED,
    NORMAL_WEIGHT_BOOST, USE_OVERSAMPLING, VAL_SPLIT,
)


def crop_margin(img_array, crop_percent=0.08):
    """
    Cắt bỏ crop_percent (ví dụ 8%) các cạnh biên (Top, Bottom, Left, Right)
    để loại bỏ ký tự 'R', 'L', viền đen/trắng xung quanh ảnh X-quang.
    Hỗ trợ cả ảnh 2D (grayscale), 3D (RGB), và 4D (batch).
    """
    if img_array.ndim == 2:
        img_array = np.expand_dims(img_array, axis=-1)

    if img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)

    h, w = img_array.shape[:2]
    top = int(h * crop_percent)
    bottom = int(h * (1.0 - crop_percent))
    left = int(w * crop_percent)
    right = int(w * (1.0 - crop_percent))

    cropped = img_array[top:bottom, left:right]
    resized = tf.image.resize(cropped, [h, w]).numpy()
    return resized


def custom_preprocessing(img_array):
    """
    Tiền xử lý kết hợp:
    1. Margin Crop cắt bỏ 8% viền nhiễu và ký tự 'R', 'L'
    2. vgg_preprocess cho VGG16
    """
    cropped_img = crop_margin(img_array, crop_percent=0.08)
    return vgg_preprocess(cropped_img)


def make_generators():
    """
    Augmentation chỉ áp dụng cho tập train.
    Train generator dùng augmentation mạnh hơn để cải thiện recall NORMAL.
    Val generator lấy 15% từ TRAIN_DIR (thay vì VAL_DIR chỉ có 16 ảnh).
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocessing,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,             # cải thiện NORMAL recall
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],  # thay đổi độ sáng
        channel_shift_range=20.0,     # dịch chuyển kênh màu
        horizontal_flip=True,
        validation_split=VAL_SPLIT,   # lấy val từ TRAIN_DIR thay vì VAL_DIR 16 ảnh
    )

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", seed=SEED, subset="training",
    )
    # Val lấy từ TRAIN_DIR (cùng datagen để dùng đúng validation_split)
    # subset="validation" không augment (ImageDataGenerator chỉ augment training subset)
    val_datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocessing,
        validation_split=VAL_SPLIT,
    )
    val_gen = val_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", shuffle=False, seed=SEED, subset="validation",
    )
    test_datagen = ImageDataGenerator(preprocessing_function=custom_preprocessing)
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", shuffle=False,
    )
    print(f"[Generators] train: {train_gen.n} ảnh | val: {val_gen.n} ảnh | test: {test_gen.n} ảnh")
    return train_gen, val_gen, test_gen


def make_oversampled_generator():
    """
    Tạo generator với oversampling lớp NORMAL:
    - Đếm số ảnh NORMAL vs PNEUMONIA trong TRAIN_DIR
    - NORMAL được lồng vào nhiều lần hơn để tỉ lệ gần bằng PNEUMONIA
    - Trả về generator tổng hợp (Python generator) + val_gen + test_gen
    Cũng trả về steps_per_epoch phù hợp với số lượng sau oversample.
    """
    # Đếm số ảnh mỗi lớp trong TRAIN_DIR
    normal_dir = os.path.join(TRAIN_DIR, "NORMAL")
    pneumonia_dir = os.path.join(TRAIN_DIR, "PNEUMONIA")
    n_normal = len([f for f in os.listdir(normal_dir)
                    if f.lower().endswith(("png", "jpg", "jpeg"))])
    n_pneumonia = len([f for f in os.listdir(pneumonia_dir)
                       if f.lower().endswith(("png", "jpg", "jpeg"))])

    print(f"[Oversample] NORMAL: {n_normal} | PNEUMONIA: {n_pneumonia}")
    oversample_ratio = math.ceil(n_pneumonia / n_normal)
    print(f"[Oversample] Ratio PNEUMONIA/NORMAL = {oversample_ratio:.2f}x → "
          f"lặp NORMAL {oversample_ratio} lần")

    # Generator riêng cho NORMAL (augmentation mạnh)
    normal_datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocessing,
        rotation_range=20,
        width_shift_range=0.12,
        height_shift_range=0.12,
        shear_range=0.15,
        zoom_range=0.2,
        brightness_range=[0.75, 1.25],
        channel_shift_range=25.0,
        horizontal_flip=True,
    )
    # Generator cho PNEUMONIA (augmentation nhẹ)
    pneumonia_datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocessing,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
    )

    # Tạo sub-generator cho từng lớp
    normal_gen = normal_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", seed=SEED, classes=["NORMAL"],
    )
    pneumonia_gen = pneumonia_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", seed=SEED, classes=["PNEUMONIA"],
    )

    def combined_generator(normal_gen, pneumonia_gen, oversample_ratio):
        """Kết hợp NORMAL (x oversample_ratio) và PNEUMONIA theo batch."""
        while True:
            # yield oversample_ratio batch NORMAL cho mỗi 1 batch PNEUMONIA
            for _ in range(oversample_ratio):
                x_n, y_n = next(normal_gen)
                # Sửa label về 0 (NORMAL = 0 theo alphabet)
                y_n = np.zeros(len(y_n), dtype=np.float32)
                yield x_n, y_n
            x_p, y_p = next(pneumonia_gen)
            # Sửa label về 1 (PNEUMONIA = 1 theo alphabet)
            y_p = np.ones(len(y_p), dtype=np.float32)
            yield x_p, y_p

    combined_gen = combined_generator(normal_gen, pneumonia_gen, oversample_ratio)

    # Tính steps_per_epoch: sau oversample, số batch NORMAL tăng lên
    total_samples_after = n_normal * oversample_ratio + n_pneumonia
    steps_per_epoch = math.ceil(total_samples_after / BATCH_SIZE)

    print(f"[Oversample] steps_per_epoch sau oversample: {steps_per_epoch}")

    # Val lấy từ TRAIN_DIR (validation_split=VAL_SPLIT) để có ~780 mẫu ổn định
    val_datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocessing,
        validation_split=VAL_SPLIT,
    )
    val_gen = val_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", shuffle=False, seed=SEED, subset="validation",
    )
    test_datagen = ImageDataGenerator(preprocessing_function=custom_preprocessing)
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", shuffle=False,
    )

    print(f"[Oversample] val: {val_gen.n} ảnh | test: {test_gen.n} ảnh")
    return combined_gen, val_gen, test_gen, steps_per_epoch


def get_class_weights(train_gen):
    """
    Xử lý mất cân bằng lớp (PNEUMONIA thường nhiều hơn NORMAL).
    Sau khi tính balanced weight, nhân thêm NORMAL_WEIGHT_BOOST cho NORMAL.
    """
    labels = train_gen.classes
    weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(labels), y=labels,
    )
    class_weight_dict = dict(enumerate(weights))
    print("Class indices:", train_gen.class_indices)
    print("Class weights (trước boost):", class_weight_dict)

    # Tìm index của NORMAL (thứ tự alphabet → NORMAL = 0)
    normal_idx = train_gen.class_indices.get("NORMAL", 0)
    class_weight_dict[normal_idx] *= NORMAL_WEIGHT_BOOST

    print(f"Class weights (sau boost NORMAL x{NORMAL_WEIGHT_BOOST}):", class_weight_dict)
    return class_weight_dict
