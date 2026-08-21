"""
Tiền xử lý dữ liệu: tạo ImageDataGenerator cho train/val/test,
tính class_weight để xử lý mất cân bằng lớp.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

from src.config import TRAIN_DIR, VAL_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE, SEED


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
    """Augmentation chỉ áp dụng cho tập train."""
    train_datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocessing,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
    )
    val_test_datagen = ImageDataGenerator(preprocessing_function=custom_preprocessing)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", seed=SEED,
    )
    val_gen = val_test_datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", shuffle=False,
    )
    test_gen = val_test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", shuffle=False,
    )
    return train_gen, val_gen, test_gen


def get_class_weights(train_gen):
    """Xử lý mất cân bằng lớp (PNEUMONIA thường nhiều hơn NORMAL)."""
    labels = train_gen.classes
    weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(labels), y=labels,
    )
    class_weight_dict = dict(enumerate(weights))
    print("Class indices:", train_gen.class_indices)
    print("Class weights:", class_weight_dict)
    return class_weight_dict
