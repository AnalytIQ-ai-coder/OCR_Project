from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "preprocessed_data" / "dataset.yaml"

def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATA_YAML),
        imgsz=640,
        epochs=50,
        batch=16,
        device="cpu",
        workers=4,
        project=str(PROJECT_ROOT / "runs"),
        name="plate_yolo",
    )

    best = PROJECT_ROOT / "runs" / "plate_yolo" / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError("Training finished but best.pt not found")

    print(f"Training finished. Best model: {best}")

if __name__ == "__main__":
    main()
