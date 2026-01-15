import sys
import time
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

from app.core.dataset import TestDataset
from app.core.metrics import calculate_iou, calculate_final_grade
from app.core.ocr import PlateOCR
from app.core.postprocess import postprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

MODEL_PATH = ROOT / "runs" / "plate_yolo" / "weights" / "best.pt"
CONF_TH = 0.25
IOU_TH = 0.5
IMG_SIZE = 640

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO(str(MODEL_PATH))

    ocr = PlateOCR(gpu=torch.cuda.is_available())

    dataset = TestDataset(ROOT)
    print(f"Test samples: {len(dataset)}")

    correct = 0
    total = 0
    iou_sum = 0.0

    t0 = time.time()

    for sample in dataset:
        image = cv2.imread(sample["image"])
        if image is None:
            continue

        gt_bbox = sample["bbox"]
        gt_text = sample["plate"]

        results = model.predict(
            image,
            conf=CONF_TH,
            imgsz=IMG_SIZE,
            device=device,
            verbose=False
        )

        if results[0].boxes is None or len(results[0].boxes) == 0:
            continue

        box = results[0].boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        pred_bbox = [x1, y1, x2, y2]

        iou = calculate_iou(gt_bbox, pred_bbox)
        iou_sum += iou

        plate = image[y1:y2, x1:x2]
        if plate.size == 0:
            continue

        raw_text = ocr.read(plate)
        pred_text = postprocess(raw_text)

        gt_norm = gt_text.replace(" ", "").upper()
        pred_norm = pred_text.replace(" ", "").upper()

        status = "OK" if (iou >= IOU_TH and gt_norm == pred_norm) else "FAIL"

        print(
            f"GT: {gt_norm} | "
            f"OCR: {pred_norm} | "
            f"{status}"
        )

        if iou >= IOU_TH:
            total += 1
            if gt_norm == pred_norm:
                correct += 1

    elapsed = time.time() - t0
    time_per_100 = elapsed / len(dataset) * 100 if len(dataset) else 0.0

    accuracy = (correct / total * 100) if total else 0.0
    avg_iou = iou_sum / total if total else 0.0
    grade = calculate_final_grade(accuracy, time_per_100)

    print("\n==============================")
    print(f"Accuracy:           {accuracy:.2f}%")
    print(f"IoU:                {avg_iou:.3f}")
    print(f"Time / 100 images:  {time_per_100:.2f} s")
    print(f"Final grade:        {grade}")
    print("==============================\n")

if __name__ == "__main__":
    main()
