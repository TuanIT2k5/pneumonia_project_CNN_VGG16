"""
Kiến trúc mô hình kết hợp: VGG16 (feature extractor) + CNN tự thiết kế.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG16


def build_combined_model(input_shape=(224, 224, 3), freeze_base=True, fine_tune_at=15):
    """
    - VGG16 (include_top=False) trích xuất đặc trưng từ ảnh X-quang.
    - Feature map đi qua các lớp Conv2D tự thiết kế để học thêm đặc trưng
      chuyên biệt cho bài toán, trước khi phân loại nhị phân.

    freeze_base=True  -> đóng băng toàn bộ VGG16 (giai đoạn 1)
    freeze_base=False -> mở khóa vài layer cuối để fine-tune (giai đoạn 2)
    """
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

    if freeze_base:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)   # feature map (7, 7, 512)

    # ----- Phần CNN tự thiết kế -----
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="custom_conv1")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="custom_conv2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="vgg16_custom_cnn")

    lr = 1e-4 if freeze_base else 1e-5
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    return model
