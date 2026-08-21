"""
Entry point chạy chương trình từ dòng lệnh.

Cách dùng:
  python main.py --mode train      # huấn luyện mô hình (2 giai đoạn)
  python main.py --mode evaluate   # đánh giá mô hình đã lưu trên tập test
"""

import argparse
import os
import ctypes as _ctypes

# Fix Python 3.8+ Windows: pre-load CUDA/cuDNN DLLs bằng ctypes trước khi import tensorflow
_venv_scripts = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv_gpu", "Scripts")
_cuda_dlls = [
    "cudart64_110.dll", "cublas64_11.dll", "cublasLt64_11.dll",
    "cufft64_10.dll", "curand64_10.dll", "cusolver64_11.dll",
    "cusparse64_11.dll", "cudnn64_8.dll",
]
for _dll in _cuda_dlls:
    _dll_path = os.path.join(_venv_scripts, _dll)
    if os.path.exists(_dll_path):
        try:
            _ctypes.CDLL(_dll_path)
        except OSError as _e:
            print(f"[DLL] Warning: {_dll}: {_e}")
    else:
        print(f"[DLL] Missing: {_dll_path}")

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

from src.config import MODELS_DIR
from src.train import run_training
from src.evaluate import evaluate_on_test, plot_training_history, load_history_json
from src.data_preprocessing import make_generators


def main():
    parser = argparse.ArgumentParser(description="Pneumonia Detection CNN Pipeline")
    parser.add_argument(
        "--mode", choices=["train", "evaluate", "app"], required=True,
        help="train: huấn luyện mô hình | evaluate: đánh giá mô hình | app: mở giao diện Streamlit AI",
    )
    args = parser.parse_args()

    if args.mode == "train":
        model, history, test_gen = run_training()
        plot_training_history(
            history, save_path=os.path.join(MODELS_DIR, "training_curves.png")
        )
        evaluate_on_test(model, test_gen)

    elif args.mode == "evaluate":
        model_path = os.path.join(MODELS_DIR, "final_model.h5")
        model = tf.keras.models.load_model(model_path)
        _, _, test_gen = make_generators()
        evaluate_on_test(model, test_gen)

    elif args.mode == "app":
        app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
        os.system(f"streamlit run {app_path}")


if __name__ == "__main__":
    main()
