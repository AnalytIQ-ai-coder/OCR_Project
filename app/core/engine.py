from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

from app.core.ocr import PlateOCR
from app.core.postprocess import postprocess

_model = None
_ocr = None

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "runs" / "plate_yolo" / "weights" / "best.pt"
CONF_TH = 0.25


def load_models():

    global _model, _ocr

    if _model is not None and _ocr is not None:
        return

    print("Loading YOLO model...")
    _model = YOLO(MODEL_PATH)

    print("Loading OCR model...")
    _ocr = PlateOCR(gpu=torch.cuda.is_available())

    print("Models loaded")


def analyze_image(image):

    if _model is None or _ocr is None:
        raise RuntimeError("Models not loaded. Call load_models() first.")

    results = _model.predict(
        image,
        conf=CONF_TH,
        imgsz=640,
        verbose=False
    )

    if results[0].boxes is None or len(results[0].boxes) == 0:
        return {"ok": False, "reason": "no_plate"}

    box = results[0].boxes.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)

    plate = image[y1:y2, x1:x2]
    if plate.size == 0:
        return {"ok": False, "reason": "empty_crop"}

    raw = _ocr.read(plate)
    final = postprocess(raw)

    if not final:
        return {"ok": False, "reason": "ocr_failed"}

    return {
        "ok": True,
        "plate": final
    }
