"""Cấu hình chung cho toàn bộ project."""

import os

# Đường dẫn dataset (điều chỉnh nếu cấu trúc thư mục khác)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

IMG_SIZE = (224, 224)   # bắt buộc 224x224 vì dùng VGG16 làm base
BATCH_SIZE = 64         # 32 → gradient updates nhiều hơn, hội tụ mượt hơn 64
SEED = 42

EPOCHS_PHASE1 = 20   # đóng băng VGG16, chỉ train phần CNN phía sau
EPOCHS_PHASE2 = 15   # tăng từ 10→15 để fine-tune kỹ hơn
FINE_TUNE_AT = 10    # mở khóa nhiều layer VGG16 hơn (từ 15→10): block4_conv* + block5_conv*

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]   # thứ tự alphabet, khớp với flow_from_directory

# --- Xử lý mất cân bằng lớp ---
NORMAL_WEIGHT_BOOST = 2.5
USE_OVERSAMPLING = True

# --- Ổn định huấn luyện ---
VAL_SPLIT = 0.25
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5    # tăng từ 3→5: kiên nhẫn hơn trước khi giảm LR
REDUCE_LR_FACTOR = 0.1    # giảm từ 0.2→0.1: LR giảm mềm hơn, tránh spike
REDUCE_LR_MIN_DELTA = 0.001

