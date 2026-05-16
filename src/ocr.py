"""
OCR module — runs Tesseract and PaddleOCR with common output format.

Each engine returns a list of dicts:
    {"text": str, "conf": float (0-1), "bbox": (x, y, w, h)}
"""

import pytesseract
from paddleocr import PaddleOCR

# Initialize PaddleOCR once at module level (avoids reloading models per call)
_paddle_engine = None


def _get_paddle_engine():
    global _paddle_engine
    if _paddle_engine is None:
        _paddle_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _paddle_engine


def run_tesseract(image):
    """
    Run Tesseract OCR.

    Args:
        image: numpy array (BGR or grayscale).

    Returns:
        List of dicts with keys: text, conf (0-1), bbox (x, y, w, h)
    """
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    results = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])

        if conf < 0 or text == "":
            continue

        results.append({
            "text": text,
            "conf": conf / 100.0,
            "bbox": (data["left"][i], data["top"][i],
                     data["width"][i], data["height"][i]),
        })

    return results


def run_paddleocr(image_path):
    """
    Run PaddleOCR.

    Args:
        image_path: path to image file (string or Path).

    Returns:
        List of dicts with keys: text, conf (0-1), bbox (x, y, w, h)
    """
    engine = _get_paddle_engine()
    raw = engine.ocr(str(image_path), cls=True)

    results = []
    for item in raw[0]:
        points = item[0]
        text = item[1][0]
        conf = float(item[1][1])

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x = int(min(xs))
        y = int(min(ys))
        w = int(max(xs) - min(xs))
        h = int(max(ys) - min(ys))

        results.append({
            "text": text,
            "conf": conf,
            "bbox": (x, y, w, h),
        })

    return results