"""
FastAPI app phục vụ dự đoán viêm phổi từ ảnh X-quang.

Chạy:
  uvicorn api.main:app --reload --port 8000

Sau đó mở http://127.0.0.1:8000/docs để test qua giao diện Swagger.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException

from api.schemas import PredictionResponse
from api.inference import predict

app = FastAPI(
    title="Pneumonia Detection API",
    description="API dự đoán viêm phổi từ ảnh X-quang ngực bằng mô hình VGG16 + CNN tự thiết kế",
    version="1.0.0",
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng file không hỗ trợ: {file.content_type}. Chỉ nhận JPEG/PNG.",
        )

    image_bytes = await file.read()
    try:
        result = predict(image_bytes)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return PredictionResponse(**result)
