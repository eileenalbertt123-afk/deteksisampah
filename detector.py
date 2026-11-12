import random

CLASSES = ["plastik", "kertas_kardus", "logam", "organik", "kaca"]

def detect(image, conf_threshold=0.25):
    """image: PIL.Image -> list of {label, confidence, bbox}"""
    w, h = image.size
    results = []
    for _ in range(random.randint(5, 12)):
        x1 = random.randint(0, int(w * 0.6))
        y1 = random.randint(0, int(h * 0.6))
        x2 = min(w, x1 + random.randint(w // 10, w // 3))
        y2 = min(h, y1 + random.randint(h // 10, h // 3))
        results.append({
            "label": random.choice(CLASSES),
            "confidence": round(random.uniform(0.2, 0.95), 2),
            "bbox": [x1, y1, x2, y2],
        })
    return [r for r in results if r["confidence"] >= conf_threshold]