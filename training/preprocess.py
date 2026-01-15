import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "preprocessed_data"

TRAIN_RATIO = 0.7
CLASS_NAME = "plate"

class YoloPreprocessor:
    def __init__(self):
        self.dataset_dir = DATASET_DIR
        self.output_dir = OUTPUT_DIR
        self.train_ratio = TRAIN_RATIO
        self.class_name = CLASS_NAME

        self.images_dir = self.dataset_dir / "photos"
        self.annotations = self.dataset_dir / "annotations.xml"

        self.images_out = self.output_dir / "images"
        self.labels_out = self.output_dir / "labels"

    def prepare(self):

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")
        if not self.annotations.exists():
            raise FileNotFoundError(f"Annotations not found: {self.annotations}")

        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        for p in [
            self.images_out / "train",
            self.images_out / "val",
            self.labels_out / "train",
            self.labels_out / "val",
        ]:
            p.mkdir(parents=True, exist_ok=True)

        samples = self._parse_xml()
        random.shuffle(samples)

        split = int(len(samples) * self.train_ratio)

        for idx, (name, boxes) in enumerate(samples):
            subset = "train" if idx < split else "val"

            src = self.images_dir / name
            dst = self.images_out / subset / name
            shutil.copy(src, dst)

            img = cv2.imread(str(src))
            if img is None:
                continue

            h, w = img.shape[:2]

            label_file = self.labels_out / subset / f"{Path(name).stem}.txt"
            with open(label_file, "w") as f:
                for box in boxes:
                    xtl = float(box.get("xtl"))
                    ytl = float(box.get("ytl"))
                    xbr = float(box.get("xbr"))
                    ybr = float(box.get("ybr"))

                    cx = ((xtl + xbr) / 2) / w
                    cy = ((ytl + ybr) / 2) / h
                    bw = (xbr - xtl) / w
                    bh = (ybr - ytl) / h

                    if 0 <= cx <= 1 and 0 <= cy <= 1:
                        f.write(f"0 {cx} {cy} {bw} {bh}\n")

        self._write_yaml()

        print("Preprocessing finished")
    def _parse_xml(self):
        tree = ET.parse(self.annotations)
        root = tree.getroot()

        samples = []

        for img in root.findall("image"):
            name = img.get("name")
            boxes = [
                b for b in img.findall("box")
                if b.get("label") == self.class_name
            ]
            if boxes:
                samples.append((name, boxes))

        if not samples:
            raise RuntimeError("No samples in annotations.xml")

        return samples

    def _write_yaml(self):
        data = {
            "path": str(self.output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": 1,
            "names": [self.class_name],
        }

        with open(self.output_dir / "dataset.yaml", "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

if __name__ == "__main__":
    YoloPreprocessor().prepare()
