"""Локальний inference готової YOLO-моделі."""
from functools import lru_cache
from io import BytesIO
from time import perf_counter

from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

WEIGHTS = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.25


class InvalidImageError(ValueError):
    """Порожній файл або некоректне зображення."""


class DetectionError(RuntimeError):
    """Помилка завантаження моделі або виконання inference."""


@lru_cache(maxsize=1)
def load_model():
    """Завантажити модель один раз за життя процесу."""
    try:
        return YOLO(WEIGHTS)
    except Exception as exc:
        raise DetectionError(
            f"Не вдалося завантажити модель {WEIGHTS}: {type(exc).__name__}"
        ) from exc


def detect(image_bytes: bytes, confidence: float = DEFAULT_CONFIDENCE) -> dict:
    if not image_bytes:
        raise InvalidImageError("Файл порожній. Завантажте зображення.")
    if not 0.0 <= confidence <= 1.0:
        raise InvalidImageError("Поріг confidence має бути від 0 до 1.")

    try:
        with Image.open(BytesIO(image_bytes)) as source:
            source.verify()
        with Image.open(BytesIO(image_bytes)) as source:
            image = source.convert("RGB")
            width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(
            "Файл не є коректним зображенням. Завантажте JPG, PNG або інше зображення."
        ) from exc

    model = load_model()
    started = perf_counter()
    try:
        results = model.predict(source=image, conf=confidence, verbose=False)
    except Exception as exc:
        raise DetectionError(
            f"Не вдалося виконати детекцію: {type(exc).__name__}"
        ) from exc
    elapsed_ms = (perf_counter() - started) * 1000

    detections = []
    result = results[0]
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            x1, y1, x2, y2 = [round(float(v), 2) for v in box.xyxy[0].tolist()]
            detections.append({
                "class_id": class_id,
                "class_name": str(result.names[class_id]),
                "confidence": round(float(box.conf.item()), 4),
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })

    return {
        "image": {"width": width, "height": height},
        "confidence_threshold": confidence,
        "count": len(detections),
        "inference_time_ms": round(elapsed_ms, 2),
        "detections": detections,
    }
