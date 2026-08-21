# Pneumonia Detection từ ảnh X-quang (VGG16 + CNN tự thiết kế)

## Cấu trúc project

```
pneumonia_project/
├── dataset/                  # Dữ liệu ảnh X-quang (tự thêm ảnh vào đây)
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
├── notebooks/
│   └── train_and_evaluate.ipynb   # EDA, tiền xử lý, train, đánh giá, biểu đồ, Grad-CAM
├── src/                       # Code lõi, dùng chung cho cả notebook lẫn CLI/API
│   ├── config.py              # Đường dẫn, hyperparameter
│   ├── data_preprocessing.py  # Data generator, augmentation, class weight
│   ├── model.py                # Kiến trúc VGG16 + CNN tự thiết kế
│   ├── train.py                # Huấn luyện 2 giai đoạn
│   ├── evaluate.py             # Confusion matrix, precision/recall/F1, ROC-AUC
│   └── gradcam.py              # Trực quan hóa Grad-CAM
├── api/                        # FastAPI phục vụ dự đoán qua HTTP
│   ├── main.py                 # FastAPI app, endpoint /predict
│   ├── inference.py            # Load model, tiền xử lý ảnh, dự đoán
│   └── schemas.py              # Pydantic response schema
├── models/                     # Mô hình đã train (.keras) — sinh ra sau khi train
├── outputs/                     # Biểu đồ, confusion matrix, classification report
├── main.py                      # Entry point CLI: train / evaluate
└── requirements.txt
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy huấn luyện (CLI)

```bash
python main.py --mode train
```

Mô hình tốt nhất sẽ được lưu vào `models/final_model.keras`, kèm biểu đồ
loss/accuracy tại `models/training_curves.png` và kết quả đánh giá tại `outputs/`.

## Chạy đánh giá lại mô hình đã train

```bash
python main.py --mode evaluate
```

## Chạy Giao diện Streamlit AI Dashboard (Khuyên dùng)

```bash
streamlit run app.py
# Hoặc:
python main.py --mode app
```

Giao diện Streamlit cao cấp chủ đề **AI Cyberpunk Dark Mode** tích hợp đầy đủ 4 tính năng:
1. **Chẩn đoán & Grad-CAM (Inference & XAI):** Upload ảnh X-quang hoặc chọn mẫu test, hiển thị kết quả chẩn đoán + bản đồ nhiệt Grad-CAM giải thích vùng chú ý của AI.
2. **Training Studio:** Huấn luyện mô hình 2 giai đoạn (Phase 1 & Phase 2) với các siêu tham số tùy chỉnh trực quan.
3. **Đánh giá & Analytics:** Đánh giá các chỉ số Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix & phân tích các ca dự đoán sai (Error Cases).
4. **Khám phá Dữ liệu (EDA & Margin Crop):** Thống kê dữ liệu & xem thử tác dụng loại bỏ viền nhiễu của Margin Crop.

## Chạy notebook (khuyến nghị dùng để làm báo cáo, vì có sẵn biểu đồ + Grad-CAM)

```bash
jupyter notebook notebooks/train_and_evaluate.ipynb
```

## Chạy API dự đoán

```bash
uvicorn api.main:app --reload --port 8000
```

Mở `http://127.0.0.1:8000/docs` để test endpoint `/predict` (upload ảnh X-quang,
nhận về nhãn dự đoán NORMAL/PNEUMONIA + xác suất).

## Dataset

Tải bộ dữ liệu "Chest X-Ray Pneumonia" trên Kaggle và giải nén vào thư mục
`dataset/` theo đúng cấu trúc `train/val/test` như trên.
