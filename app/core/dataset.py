import xml.etree.ElementTree as ET
from pathlib import Path

class TestDataset:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)

        self.annotations = self.project_root / "dataset" / "annotations.xml"
        self.images_dir = self.project_root / "preprocessed_data" / "images" / "val"

        if not self.annotations.exists():
            raise FileNotFoundError(f"Missing annotations: {self.annotations}")

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Missing images dir: {self.images_dir}")

        self.samples = self._load()

    def _load(self):
        tree = ET.parse(self.annotations)
        root = tree.getroot()

        samples = []

        for img in root.findall("image"):
            name = img.attrib["name"]
            img_path = self.images_dir / name

            if not img_path.exists():
                continue

            box = img.find("box")
            if box is None:
                continue

            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])

            attr = box.find("attribute")
            plate_text = attr.text.strip().upper()

            samples.append({
                "image": str(img_path),
                "bbox": [xtl, ytl, xbr, ybr],
                "plate": plate_text
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)
