"""
Grad-CAM: trực quan hóa vùng ảnh X-quang mà mô hình "chú ý" khi ra quyết định.
Áp dụng lên layer Conv2D cuối cùng trong phần CNN tự thiết kế (custom_conv2).
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="custom_conv2"):
    """
    img_array: ảnh đã tiền xử lý, shape (1, H, W, 3)
    Trả về heatmap 2D giá trị [0, 1]
    """
    inputs = model.input if hasattr(model, "input") else model.inputs
    grad_model = tf.keras.models.Model(
        inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, 0]   # binary classification, 1 output node

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(img_input, heatmap, alpha=0.4, img_size=(224, 224)):
    """
    Chồng heatmap lên ảnh gốc để hiển thị.
    img_input: đường dẫn file (str), hoặc PIL Image, hoặc np.ndarray
    """
    from PIL import Image
    if isinstance(img_input, str):
        img = keras_image.load_img(img_input, target_size=img_size)
        img = keras_image.img_to_array(img)
    elif isinstance(img_input, Image.Image):
        img = img_input.convert("RGB").resize(img_size)
        img = keras_image.img_to_array(img)
    elif isinstance(img_input, np.ndarray):
        img = img_input.copy()
        if img.dtype != np.uint8 and img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
    else:
        raise ValueError("Unsupported img_input type for overlay_gradcam")

    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras_image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_size[1], img_size[0]))
    jet_heatmap = keras_image.img_to_array(jet_heatmap)

    superimposed = jet_heatmap * alpha + img
    superimposed = np.clip(superimposed, 0, 255)
    superimposed = keras_image.array_to_img(superimposed)
    return superimposed


def visualize_gradcam(img_path, model, last_conv_layer_name="custom_conv2", img_size=(224, 224)):
    """Hàm tiện ích: load ảnh, tính Grad-CAM, hiển thị ảnh gốc + heatmap chồng lên."""
    from src.data_preprocessing import crop_margin
    img = keras_image.load_img(img_path, target_size=img_size)
    array = keras_image.img_to_array(img)
    array = crop_margin(array, crop_percent=0.08)
    array = np.expand_dims(array, axis=0)
    array = vgg_preprocess(array)

    heatmap = make_gradcam_heatmap(array, model, last_conv_layer_name)
    superimposed = overlay_gradcam(img_path, heatmap, img_size=img_size)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(keras_image.load_img(img_path, target_size=img_size))
    axes[0].set_title("Ảnh gốc")
    axes[0].axis("off")

    axes[1].imshow(superimposed)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
