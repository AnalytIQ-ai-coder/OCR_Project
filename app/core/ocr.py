import re
import cv2
import numpy as np
import easyocr

class PlateOCR:
    def __init__(self, gpu: bool = False):
        self.reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)
        self.allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def _remove_blue_strip(self, img: np.ndarray) -> np.ndarray:
        if img.size == 0:
            return img

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]

        lower_blue = np.array([90, 60, 60])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        scan_width = int(w * 0.30)
        max_cut = int(w * 0.20)

        cut_x = 0
        in_blue = False

        for x in range(scan_width):
            blue_ratio = np.count_nonzero(mask[:, x]) / h
            if blue_ratio > 0.35:
                in_blue = True
                cut_x = x
            elif in_blue:
                break

        cut_x = min(cut_x, max_cut)

        if cut_x > 0:
            return img[:, cut_x + 2:]

        return img

    def read(self, plate_bgr: np.ndarray) -> str:
        if plate_bgr is None or plate_bgr.size == 0:
            return ""

        h, w = plate_bgr.shape[:2]

        margin = 0.02
        plate = plate_bgr[
            int(h * margin): int(h * (1 - margin)),
            int(w * margin): int(w * (1 - margin)),
        ]

        plate = self._remove_blue_strip(plate)

        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        if gray.shape[0] < 60:
            gray = cv2.resize(
                gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
            )

        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        inv = cv2.bitwise_not(binary)
        contours, _ = cv2.findContours(
            inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        ih, iw = binary.shape[:2]

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)

            if ch > ih * 0.85 and cw < iw * 0.08:
                cv2.drawContours(binary, [cnt], -1, 255, -1)

            elif cw > iw * 0.45:
                cv2.drawContours(binary, [cnt], -1, 255, -1)

            elif ch < ih * 0.30:
                cv2.drawContours(binary, [cnt], -1, 255, -1)

        kernel = np.ones((2, 1), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)

        padded = cv2.copyMakeBorder(
            binary, 10, 10, 10, 10,
            cv2.BORDER_CONSTANT, value=255
        )

        results = self.reader.readtext(
            padded,
            detail=0,
            allowlist=self.allowlist
        )

        text = "".join(results)
        text = re.sub(r"[^A-Z0-9]", "", text)

        if len(text) < 5 or len(text) > 8:
            return ""

        return text
