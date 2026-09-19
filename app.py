"""
Pneumonia Detection - Futuristic AI Dashboard (Streamlit)
Giao diện chẩn đoán, huấn luyện & đánh giá mô hình phân loại viêm phổi từ ảnh X-quang.
"""

import os
import io
import sys
import random
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# ── GPU Fix: inject CUDA paths vào process PATH trước khi import tensorflow ──
_cuda_candidates = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv_gpu", "Scripts"),
]
for _p in _cuda_candidates:
    if os.path.isdir(_p) and _p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(_p)
        except Exception:
            pass

# Force legacy keras
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf

# Kích hoạt Memory Growth để tránh lỗi OOM (Out Of Memory) trên GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Memory growth config error: {e}")

from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

# Đảm bảo import được các module trong src/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.config import (
    DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    MODELS_DIR, OUTPUTS_DIR, IMG_SIZE, BATCH_SIZE,
    CLASS_NAMES, EPOCHS_PHASE1, EPOCHS_PHASE2
)
from src.data_preprocessing import crop_margin, make_generators, get_class_weights
from src.model import build_combined_model
from src.train import run_training
from src.evaluate import evaluate_on_test, plot_training_history, load_history_json
from src.gradcam import make_gradcam_heatmap, overlay_gradcam

# ---------------------------------------------------------
# Page Configuration & AI Cyberpunk Theme CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="PneumoVision AI - Diagnostic & Studio",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Futuristic AI Theme (Glassmorphism & Dark Mode)
st.markdown("""
<style>
    /* Dark AI Theme Background */
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #0e1424 50%, #060911 100%);
        color: #e0e6ed;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    /* Header Styling */
    .ai-header {
        background: linear-gradient(90deg, rgba(0, 242, 254, 0.15) 0%, rgba(79, 172, 254, 0.05) 100%);
        border: 1px solid rgba(0, 242, 254, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 242, 254, 0.1);
    }
    
    .ai-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .ai-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 6px;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(18, 24, 40, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(0, 242, 254, 0.4);
    }
    
    /* Custom Badges */
    .badge-normal {
        background: rgba(0, 230, 118, 0.15);
        color: #00e676;
        border: 1px solid rgba(0, 230, 118, 0.4);
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    
    .badge-pneumonia {
        background: rgba(255, 23, 68, 0.15);
        color: #ff1744;
        border: 1px solid rgba(255, 23, 68, 0.4);
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    
    /* Custom Metrics */
    .metric-container {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #00f2fe;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Hide default streamlit menu decoration */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Helper Functions & Model Caching
# ---------------------------------------------------------
@st.cache_resource
def load_pneumonia_model():
    # Tìm model .keras hoặc .h5 mới nhất, bao gồm cả thư mục con
    candidate_paths = [
        os.path.join(MODELS_DIR, "final_model.keras"),
        os.path.join(MODELS_DIR, "final_model.h5"),
        os.path.join(os.path.dirname(MODELS_DIR), "pneumonia_project", "models", "final_model.keras"),
        os.path.join(os.path.dirname(MODELS_DIR), "pneumonia_project", "models", "final_model.h5"),
    ]
    best_path = None
    best_mtime = 0
    for p in candidate_paths:
        if os.path.exists(p):
            mtime = os.path.getmtime(p)
            if mtime > best_mtime:
                best_mtime = mtime
                best_path = p

    if best_path:
        try:
            model = tf.keras.models.load_model(best_path)
            return model, best_path
        except Exception as e:
            st.error(f"Lỗi khi load mô hình: {e}")
            return None, None
    return None, None


def _get_outputs_dir_for_model(model_path):
    """Trả về thư mục outputs/ tương ứng với model_path đang được dùng."""
    model_models_dir = os.path.dirname(model_path)
    base = os.path.dirname(model_models_dir)
    return os.path.join(base, "outputs")


def get_optimal_threshold():
    """Load ngưỡng tối ưu Youden's J đã lưu từ lần evaluate gần nhất."""
    npz_path = os.path.join(OUTPUTS_DIR, "raw_predictions.npz")
    if os.path.exists(npz_path):
        try:
            data = np.load(npz_path)
            fpr, tpr = data["fpr"], data["tpr"]
            J = tpr - fpr
            idx = np.argmax(J)
            # roc_thresholds không lưu trực tiếp — dùng prob threshold tính gần đúng
        except Exception:
            pass
    return 0.5


def get_image_prediction(img_pil, model, threshold=0.5):
    """Tiền xử lý ảnh (crop 8% margin + VGG preprocess) và đưa ra dự đoán."""
    img_rgb = img_pil.convert("RGB")
    w, h = img_rgb.size

    # Margin crop 8%
    crop_percent = 0.08
    crop_box = (
        int(w * crop_percent),
        int(h * crop_percent),
        int(w * (1.0 - crop_percent)),
        int(h * (1.0 - crop_percent))
    )
    img_cropped = img_rgb.crop(crop_box)
    img_resized = img_cropped.resize(IMG_SIZE)

    arr = np.array(img_resized, dtype=np.float32)
    arr_expanded = np.expand_dims(arr, axis=0)
    arr_preprocessed = vgg_preprocess(arr_expanded.copy())

    prob = float(model.predict(arr_preprocessed, verbose=0)[0][0])

    label = "PNEUMONIA" if prob >= threshold else "NORMAL"
    confidence = prob if prob >= threshold else (1.0 - prob)

    return {
        "label": label,
        "prob": prob,
        "confidence": confidence,
        "threshold": threshold,
        "img_cropped": img_cropped,
        "arr_preprocessed": arr_preprocessed
    }


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 10px 0;'>
        <h2 style='color: #00f2fe; margin:0; font-size: 1.6rem;'>🩺 PNEUMO-VISION</h2>
        <p style='color: #64748b; font-size: 0.85rem; margin-top: 4px;'>Deep Learning Clinical Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    selected_nav = st.radio(
        "Điều hướng hệ thống:",
        [
            "🩺 Chẩn đoán & Grad-CAM (Inference)",
            "📈 Lịch sử & Kết quả Huấn luyện",
            "📊 Đánh giá & Analytics (Evaluation)",
            "📂 Khám phá dữ liệu (Dataset EDA)"
        ],
        index=0
    )

    st.divider()
    
    # Model Status Card in Sidebar
    model, model_path = load_pneumonia_model()
    if model is not None:
        st.markdown(f"""
        <div style='background: rgba(0, 230, 118, 0.1); border: 1px solid rgba(0, 230, 118, 0.3); padding: 12px; border-radius: 10px;'>
            <span style='color: #00e676; font-weight: bold;'>● Model Active</span><br/>
            <small style='color: #94a3b8;'>VGG16 + Custom CNN Head</small><br/>
            <small style='color: #64748b;'>{os.path.basename(model_path)}</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background: rgba(255, 23, 68, 0.1); border: 1px solid rgba(255, 23, 68, 0.3); padding: 12px; border-radius: 10px;'>
            <span style='color: #ff1744; font-weight: bold;'>● No Saved Model Found</span><br/>
            <small style='color: #94a3b8;'>Vui lòng huấn luyện mô hình qua Jupyter Notebook.</small>
        </div>
        """, unsafe_allow_html=True)

    # Nút reload model từ disk (dùng sau khi train xong từ notebook)
    st.divider()
    if st.button("🔁 Tải Lại Model Mới Nhất", use_container_width=True,
                 help="Xóa cache RAM và load lại model mới nhất từ disk (dùng sau khi train từ notebook)"):
        st.cache_resource.clear()
        st.success("✅ Đã xóa cache! Model sẽ được load lại ở lần thao tác tiếp theo.")
        st.rerun()

    # GPU Status Card
    st.divider()
    _gpus = tf.config.list_physical_devices('GPU')
    if _gpus:
        _gpu_names = [g.name.split(":")[-1] for g in _gpus]
        st.markdown(f"""
        <div style='background: rgba(0, 242, 254, 0.08); border: 1px solid rgba(0, 242, 254, 0.35); padding: 12px; border-radius: 10px;'>
            <span style='color: #00f2fe; font-weight: bold;'>⚡ GPU Active ({len(_gpus)} device)</span><br/>
            <small style='color: #94a3b8;'>{'  |  '.join(_gpu_names)}</small><br/>
            <small style='color: #64748b;'>Training & Inference on GPU</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background: rgba(255, 165, 0, 0.08); border: 1px solid rgba(255, 165, 0, 0.3); padding: 12px; border-radius: 10px;'>
            <span style='color: #ffa500; font-weight: bold;'>🖥️ CPU Mode</span><br/>
            <small style='color: #94a3b8;'>Không tìm thấy GPU — đang dùng CPU</small><br/>
            <small style='color: #64748b;'>Xem hướng dẫn cài CUDA để kích hoạt GPU</small>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------
# Page 1: 🩺 Chẩn đoán & Grad-CAM (Inference & XAI)
# ---------------------------------------------------------
if "Chẩn đoán" in selected_nav:
    st.markdown("""
    <div class='ai-header'>
        <h1 class='ai-title'>🩺 Chẩn Đoán Ảnh X-quang Ngực & Trực Quan Hóa Grad-CAM</h1>
        <p class='ai-subtitle'>Tải lên ảnh X-quang hoặc chọn mẫu test để mô hình phân loại và giải thích vùng chú ý lâm sàng (Explainable AI).</p>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.warning("⚠️ Không tìm thấy file mô hình `models/final_model...`. Hãy chạy huấn luyện qua Jupyter Notebook trước!")
    else:
        col_input, col_result = st.columns([1, 1], gap="large")

        with col_input:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("📥 Chọn Đầu Vào Nguồn Ảnh")
            
            input_mode = st.radio("Phương thức chọn ảnh:", ["Tải lên từ máy tính (Upload)", "Chọn mẫu ngẫu nhiên từ tập Test"], horizontal=True)

            selected_img = None
            sample_img_name = ""

            if input_mode == "Tải lên từ máy tính (Upload)":
                uploaded_file = st.file_uploader("Kéo thả hoặc chọn ảnh X-quang (.png, .jpg, .jpeg):", type=["png", "jpg", "jpeg"])
                if uploaded_file is not None:
                    selected_img = Image.open(uploaded_file)
                    sample_img_name = uploaded_file.name
            else:
                test_pn_dir = os.path.join(TEST_DIR, "PNEUMONIA")
                test_norm_dir = os.path.join(TEST_DIR, "NORMAL")
                
                class_choice = st.selectbox("Chọn lớp ảnh mẫu:", ["PNEUMONIA (Viêm phổi)", "NORMAL (Bình thường)"])
                target_dir = test_pn_dir if "PNEUMONIA" in class_choice else test_norm_dir

                if os.path.exists(target_dir):
                    valid_files = [f for f in os.listdir(target_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
                    if valid_files:
                        chosen_file = st.selectbox("Chọn ảnh trong tập test:", valid_files[:20])
                        if st.button("🎲 Đổi ảnh ngẫu nhiên", use_container_width=True):
                            chosen_file = random.choice(valid_files)
                        
                        img_full_path = os.path.join(target_dir, chosen_file)
                        selected_img = Image.open(img_full_path)
                        sample_img_name = chosen_file

            if selected_img is not None:
                st.image(selected_img, caption=f"Ảnh đầu vào: {sample_img_name}", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_result:
            if selected_img is not None:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.subheader("🔬 Kết Quả Phân Tích & Chẩn Đoán AI")
                
                with st.spinner("AI đang phân tích ảnh & tính toán Grad-CAM..."):
                    pred_res = get_image_prediction(selected_img, model)
                    
                    label = pred_res["label"]
                    prob = pred_res["prob"]
                    confidence = pred_res["confidence"]
                    img_cropped = pred_res["img_cropped"]
                    arr_preprocessed = pred_res["arr_preprocessed"]

                # Diagnostic Banner
                if label == "PNEUMONIA":
                    st.markdown(f"""
                    <div style='text-align: center; margin: 15px 0;'>
                        <div class='badge-pneumonia'>⚠️ PHÁT HIỆN DẤU HIỆU VIÊM PHỔI (PNEUMONIA)</div>
                        <h2 style='color: #ff1744; margin-top: 10px;'>Độ tin cậy: {confidence*100:.2f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='text-align: center; margin: 15px 0;'>
                        <div class='badge-normal'>✅ LỒNG NGỰC BÌNH THƯỜNG (NORMAL)</div>
                        <h2 style='color: #00e676; margin-top: 10px;'>Độ tin cậy: {confidence*100:.2f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)

                # Confidence Bar
                st.markdown(f"**Xác suất mắc Viêm Phổi (Pneumonia Probability): {prob*100:.2f}%**")
                st.progress(float(prob))

                st.divider()

                # Grad-CAM Heatmap
                st.subheader("🔥 Trực Quan Hóa Vùng Chú Ý Grad-CAM")
                st.caption("Grad-CAM chiếu bản đồ nhiệt lên ảnh gốc để hiển thị vùng lồng ngực kích hoạt quyết định của AI.")

                alpha = st.slider("Độ trong suốt bản đồ nhiệt (Heatmap Alpha):", 0.1, 0.9, 0.4, 0.05)

                try:
                    heatmap = make_gradcam_heatmap(arr_preprocessed, model, last_conv_layer_name="block5_conv3")
                    superimposed_img = overlay_gradcam(img_cropped, heatmap, alpha=alpha, img_size=IMG_SIZE)

                    cam_col1, cam_col2 = st.columns(2)
                    with cam_col1:
                        st.image(img_cropped, caption="Ảnh sau Margin Crop (8%)", use_container_width=True)
                    with cam_col2:
                        st.image(superimposed_img, caption="Ảnh Grad-CAM Overlay", use_container_width=True)
                except Exception as ex:
                    st.error(f"Không thể tạo bản đồ Grad-CAM: {ex}")

                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("👈 Hãy chọn hoặc tải lên một ảnh X-quang ở bảng bên trái để bắt đầu chẩn đoán.")


# ---------------------------------------------------------
# Page 2: 📈 Lịch sử & Kết quả Huấn luyện
# ---------------------------------------------------------
elif "Lịch sử" in selected_nav:
    st.markdown("""
    <div class='ai-header'>
        <h1 class='ai-title'>📈 Lịch Sử & Kết Quả Huấn Luyện (Notebook Results)</h1>
        <p class='ai-subtitle'>Hiển thị các biểu đồ và báo cáo đánh giá được sinh ra từ quá trình huấn luyện trong Jupyter Notebook.</p>
    </div>
    """, unsafe_allow_html=True)

    if model_path is None:
        st.warning("⚠️ Chưa có mô hình nào được load. Vui lòng huấn luyện mô hình trước qua Notebook!")
    else:
        active_outputs_dir = _get_outputs_dir_for_model(model_path)
        active_models_dir = os.path.dirname(model_path)
        
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Khu vực hiển thị kết quả tĩnh từ Notebook")
        st.info(f"Đang đọc dữ liệu từ: `{active_models_dir}` và `{active_outputs_dir}`")
        
        plot_path = os.path.join(active_models_dir, "training_curves.png")
        if os.path.exists(plot_path):
            st.markdown("### Biểu đồ Loss & Accuracy")
            from PIL import Image as PILImage
            st.image(PILImage.open(plot_path), use_container_width=True)
        else:
            st.write("Chưa tìm thấy biểu đồ `training_curves.png`.")
            
        report_path = os.path.join(active_outputs_dir, "classification_report.txt")
        if os.path.exists(report_path):
            st.markdown("### Báo Cáo Phân Loại (Classification Report)")
            with open(report_path, "r", encoding="utf-8") as f:
                st.text(f.read())
                
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Page 3: 📊 Đánh giá & Analytics (Evaluation Dashboard)
# ---------------------------------------------------------
elif "Đánh giá" in selected_nav:
    st.markdown("""
    <div class='ai-header'>
        <h1 class='ai-title'>📊 Đánh Giá Mô Hình & Phân Tích Lỗi (Evaluation & Error Analysis)</h1>
        <p class='ai-subtitle'>Đánh giá hiệu năng tổng thể của mô hình trên tập dữ liệu Test độc lập (Confusion Matrix, ROC-AUC, Precision/Recall).</p>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.warning("⚠️ Chưa có mô hình nào được load. Vui lòng huấn luyện mô hình trước!")
    else:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.info("💡 Nếu vừa train xong từ notebook, nhấn **'🔁 Tải Lại Model Mới Nhất'** ở sidebar trước, rồi mới đánh giá.")
        if st.button("🔄 Thực Thi Đánh Giá Trên Tập Test", type="primary"):
            # Xóa cache và load lại model mới nhất
            st.cache_resource.clear()
            fresh_model, fresh_model_path = load_pneumonia_model()
            if fresh_model is None:
                st.error("Không tìm thấy file model. Hãy train trước!")
            else:
                active_outputs_dir = _get_outputs_dir_for_model(fresh_model_path)
                os.makedirs(active_outputs_dir, exist_ok=True)
                with st.spinner(f"Đang dùng model tại '{fresh_model_path}'..."):
                    _, _, test_gen = make_generators()
                    eval_res = evaluate_on_test(fresh_model, test_gen, save_dir=active_outputs_dir, show=False)
                    st.session_state["eval_res"] = eval_res
                    st.session_state["test_gen"] = test_gen
                    st.session_state["active_outputs_dir"] = active_outputs_dir
                    st.session_state["optimal_threshold"] = float(eval_res["optimal_threshold"])
                    st.success(f"✅ Đánh giá hoàn tất! (Model: {fresh_model_path}) — Threshold tối ưu: {eval_res['optimal_threshold']:.4f}")
        st.markdown("</div>", unsafe_allow_html=True)

        if "eval_res" in st.session_state:
            res = st.session_state["eval_res"]
            test_gen = st.session_state["test_gen"]
            
            y_true = res["y_true"]
            y_pred = res["y_pred"]
            y_prob = res["y_prob"]
            auc_score = res["auc"]

            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score as f1_score_fn
            acc  = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec  = recall_score(y_true, y_pred, zero_division=0)
            f1   = f1_score_fn(y_true, y_pred, zero_division=0)

            # Key Metrics Display
            st.markdown("### 🏆 Các Chỉ Số Cốt Lõi (Performance Metrics)")
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown(f"<div class='metric-container'><div class='metric-val'>{acc*100:.1f}%</div><div class='metric-lbl'>Accuracy</div></div>", unsafe_allow_html=True)
            with m2:
                st.markdown(f"<div class='metric-container'><div class='metric-val'>{prec*100:.1f}%</div><div class='metric-lbl'>Precision</div></div>", unsafe_allow_html=True)
            with m3:
                st.markdown(f"<div class='metric-container'><div class='metric-val'>{rec*100:.1f}%</div><div class='metric-lbl'>Recall (Sens)</div></div>", unsafe_allow_html=True)
            with m4:
                st.markdown(f"<div class='metric-container'><div class='metric-val'>{f1*100:.1f}%</div><div class='metric-lbl'>F1-Score</div></div>", unsafe_allow_html=True)
            with m5:
                st.markdown(f"<div class='metric-container'><div class='metric-val'>{auc_score:.3f}</div><div class='metric-lbl'>ROC-AUC</div></div>", unsafe_allow_html=True)


            st.divider()

            # Dùng outputs_dir tương ứng với model đã load
            _out_dir = st.session_state.get("active_outputs_dir", _get_outputs_dir_for_model(model_path) if model_path else OUTPUTS_DIR)

            col_cm, col_roc = st.columns(2)
            with col_cm:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.subheader("📌 Ma Trận Nhầm Lẫn (Confusion Matrix)")
                cm_img = os.path.join(_out_dir, "confusion_matrix.png")
                if os.path.exists(cm_img):
                    from PIL import Image as PILImage
                    st.image(PILImage.open(cm_img), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_roc:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.subheader("📈 Đường Cong ROC (ROC Curve)")
                roc_img = os.path.join(_out_dir, "roc_curve.png")
                if os.path.exists(roc_img):
                    from PIL import Image as PILImage
                    st.image(PILImage.open(roc_img), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Error Analysis Section
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("🔍 Phân Tích Ca Dự Đoán Sai (Error Analysis)")
            errors_idx = np.where(y_pred != y_true)[0]
            st.write(f"**Tổng số mẫu bị dự đoán sai:** {len(errors_idx)} / {len(y_true)} mẫu")

            if len(errors_idx) > 0:
                filenames = test_gen.filenames
                num_display = min(6, len(errors_idx))
                cols = st.columns(3)

                for i, err_idx in enumerate(errors_idx[:num_display]):
                    rel_path = filenames[err_idx]
                    full_path = os.path.join(TEST_DIR, rel_path)
                    
                    true_lbl = CLASS_NAMES[y_true[err_idx]]
                    pred_lbl = CLASS_NAMES[y_pred[err_idx]]
                    prob_val = y_prob[err_idx]

                    with cols[i % 3]:
                        if os.path.exists(full_path):
                            st.image(full_path, caption=f"Thực tế: {true_lbl} | Dự đoán: {pred_lbl} ({prob_val:.2f})", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Page 4: 📂 Khám phá dữ liệu (Dataset EDA & Preprocessing)
# ---------------------------------------------------------
elif "Khám phá" in selected_nav:
    st.markdown("""
    <div class='ai-header'>
        <h1 class='ai-title'>📂 Khám Phá Dữ Liệu & Kiểm Tra Margin Crop (EDA Inspector)</h1>
        <p class='ai-subtitle'>Thống kê phân bố tập dữ liệu X-quang và minh họa trực quan kỹ thuật cắt viền (Margin Crop 8%) loại bỏ nhiễu shortcut learning.</p>
    </div>
    """, unsafe_allow_html=True)

    col_stats, col_crop = st.columns([1, 1], gap="large")

    with col_stats:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📊 Thống Kê Phân Bố Mẫu Dữ Liệu")

        data_counts = []
        for split_name, split_dir in [("Train", TRAIN_DIR), ("Val", VAL_DIR), ("Test", TEST_DIR)]:
            for cls in CLASS_NAMES:
                cls_dir = os.path.join(split_dir, cls)
                if os.path.exists(cls_dir):
                    count = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
                    data_counts.append({"Split": split_name, "Class": cls, "Count": count})

        df_stats = pd.DataFrame(data_counts)
        st.dataframe(df_stats, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=df_stats, x="Split", y="Count", hue="Class", palette=["#00e676", "#ff1744"], ax=ax)
        ax.set_title("Phân bố số lượng mẫu theo Tập dữ liệu & Nhãn")
        ax.set_ylabel("Số lượng ảnh")
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_crop:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("✂️ Kiểm Tra Tác Dụng Margin Crop (Eliminate Artifacts)")
        st.caption("Điều chỉnh % cắt viền để thấy cách thuật toán loại bỏ các ký tự 'R'/'L' & viền đen nhiễu.")

        crop_slider = st.slider("Tỷ lệ Margin Crop (%):", 0.0, 0.20, 0.08, 0.01)

        pneumonia_train_dir = os.path.join(TRAIN_DIR, "PNEUMONIA")
        valid_imgs = [f for f in os.listdir(pneumonia_train_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

        if valid_imgs:
            sample_file = valid_imgs[0]
            img_path = os.path.join(pneumonia_train_dir, sample_file)
            
            orig_img = Image.open(img_path).resize(IMG_SIZE)
            img_arr = np.array(orig_img)
            
            cropped_arr = crop_margin(img_arr, crop_percent=crop_slider)

            c1, c2 = st.columns(2)
            with c1:
                st.image(orig_img, caption="Ảnh gốc (Có viền nhiễu)", use_container_width=True)
            with c2:
                st.image(cropped_arr.astype(np.uint8), caption=f"Sau Margin Crop ({crop_slider*100:.0f}%)", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
