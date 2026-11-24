"""FastAPI веб-рівень помічника підтримки."""
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from . import llm

app = FastAPI(title='Помічник служби підтримки — ПР3')
ROOT = Path(__file__).resolve().parent.parent
INDEX_PAGE = ROOT / 'app' / 'templates' / 'index.html'
CONTEXT_FILE = ROOT / 'context.md'

class Question(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


def load_context() -> str:
    return CONTEXT_FILE.read_text(encoding='utf-8')

@app.get('/', response_class=HTMLResponse)
def index() -> str:
    return INDEX_PAGE.read_text(encoding='utf-8')

@app.post('/api/ask')
def api_ask(payload: Question):
    if not payload.question.strip():
        raise HTTPException(status_code=422, detail='Введіть текст звернення.')
    try:
        return llm.ask(payload.question, load_context())
    except llm.LLMError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=503, detail='Не вдалося прочитати правила магазину.') from exc
