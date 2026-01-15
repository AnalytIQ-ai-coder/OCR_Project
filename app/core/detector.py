from ultralytics import YOLO
import numpy as np

class PlateDetector:
    def __init__(self, model_path: str, conf: float = 0.4):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, image: np.ndarray):

        results = self.model(image, conf=self.conf, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return None

        best = max(boxes, key=lambda b: float(b.conf))
        return tuple(map(int, best.xyxy[0]))
