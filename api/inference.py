"""Load mô hình đã train và xử lý dự đoán cho ảnh upload qua API."""

import io
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

from src.config import MODELS_DIR, IMG_SIZE

MODEL_PATH = os.path.join(MODELS_DIR, "final_model.h5")

_model = None  # lazy load, chỉ load 1 lần khi có request đầu tiên


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Không tìm thấy mô hình tại {MODEL_PATH}. "
                "Hãy chạy `python main.py --mode train` trước."
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Chuyển bytes ảnh upload thành mảng đã tiền xử lý (cắt 8% viền nhiễu) sẵn cho mô hình."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    crop_percent = 0.08
    crop_box = (
        int(w * crop_percent),
        int(h * crop_percent),
        int(w * (1.0 - crop_percent)),
        int(h * (1.0 - crop_percent))
    )
    img = img.crop(crop_box)
    img = img.resize(IMG_SIZE)
    array = np.array(img, dtype=np.float32)
    array = np.expand_dims(array, axis=0)
    array = vgg_preprocess(array)
    return array


def predict(image_bytes: bytes) -> dict:
    model = get_model()
    array = preprocess_image_bytes(image_bytes)
    prob_pneumonia = float(model.predict(array, verbose=0)[0][0])

    if prob_pneumonia >= 0.5:
        label = "PNEUMONIA"
        confidence = prob_pneumonia
    else:
        label = "NORMAL"
        confidence = 1.0 - prob_pneumonia

    return {
        "label": label,
        "probability": prob_pneumonia,
        "confidence": confidence,
    }
