"""Pydantic schemas cho request/response của API."""

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    label: str            # "NORMAL" hoặc "PNEUMONIA"
    probability: float    # xác suất PNEUMONIA (0.0 - 1.0)
    confidence: float     # độ tin cậy của nhãn được chọn
