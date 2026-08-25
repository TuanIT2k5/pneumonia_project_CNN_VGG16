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
BATCH_SIZE = 16
SEED = 42

EPOCHS_PHASE1 = 20   # đóng băng VGG16, chỉ train phần CNN phía sau (nếu dùng)
EPOCHS_PHASE2 = 5    # mở khóa vài layer cuối VGG16, fine-tune với LR nhỏ
FINE_TUNE_AT = 15    # số layer đầu của VGG16 giữ đóng băng khi fine-tune

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]   # thứ tự alphabet, khớp với flow_from_directory

# --- Xử lý mất cân bằng lớp ---
# Nhân thêm vào class weight của NORMAL sau khi sklearn tính balanced (1.0 = không tăng)
NORMAL_WEIGHT_BOOST = 2.0
# Bật oversample NORMAL để tỉ lệ gần bằng PNEUMONIA trong train_gen
USE_OVERSAMPLING = False

# --- Ổn định huấn luyện ---
# Tỉ lệ validation lấy từ TRAIN_DIR (thay vì VAL_DIR chỉ có 16 ảnh)
VAL_SPLIT = 0.15
# EarlyStopping: dừng sớm nếu val_loss không cải thiện sau bao nhiêu epoch
EARLY_STOPPING_PATIENCE = 8
# ReduceLROnPlateau: giảm LR sau bao nhiêu epoch không cải thiện
REDUCE_LR_PATIENCE = 4
# Hệ số giảm LR (LR mới = LR cũ × factor)
REDUCE_LR_FACTOR = 0.3
# Ngưỡng cải thiện tối thiểu để coi là "có tiến bộ"
REDUCE_LR_MIN_DELTA = 0.001
