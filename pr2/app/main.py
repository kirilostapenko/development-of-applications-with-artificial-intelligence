"""FastAPI веб-рівень для локальної детекції об'єктів."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from . import detector

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ваги завантажуються один раз під час старту сервера.
    detector.load_model()
    yield


app = FastAPI(
    title="Детекція об'єктів — ПР2",
    description="Локальна детекція об'єктів pretrained-моделлю YOLOv8n.",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/detect")
async def api_detect(
    image: UploadFile = File(...),
    confidence: float = Form(detector.DEFAULT_CONFIDENCE),
):
    if not 0.0 <= confidence <= 1.0:
        raise HTTPException(status_code=422, detail="Поріг confidence має бути від 0 до 1.")

    content = await image.read()
    await image.close()
    if not content:
        raise HTTPException(status_code=400, detail="Завантажений файл порожній.")

    try:
        return detector.detect(content, confidence)
    except detector.InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except detector.DetectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Внутрішня помилка під час обробки зображення.",
        ) from exc
