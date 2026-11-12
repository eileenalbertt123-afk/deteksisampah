from pathlib import Path
from ultralytics import YOLO

CLASSES = ["plastik", "kertas_kardus", "logam", "organik", "kaca"]
MODEL_PATH = Path(__file__).parent / "best.pt"

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = YOLO(str(MODEL_PATH))
    return _model

def detect(image, conf_threshold=0.25):
    """image: PIL.Image -> list of {label, confidence, bbox}"""
    model = _get_model()
    result = model.predict(image, conf=conf_threshold, max_det=300, verbose=False)[0]
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        detections.append({
            "label": CLASSES[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 2),
            "bbox": [x1, y1, x2, y2],
        })
    return detections