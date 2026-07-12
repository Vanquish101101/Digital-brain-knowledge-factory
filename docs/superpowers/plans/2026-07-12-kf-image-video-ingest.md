# Расширение kf.py под картинки и видео — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Научить `kf.py ingest` автоматически индексировать картинки (OCR + опциональное
описание сцены) и видео (транскрипт речи + разбор кадров), используя тот же
пайплайн chunk→embed→store, что уже работает для текстовых форматов.

**Architecture:** Гибрид: локальный Tesseract OCR как основной путь для картинок,
OpenRouter vision-модель — добор только когда локального текста мало. Видео: звук
через локальный faster-whisper, кадры — сэмплируются через ffmpeg и проходят ту же
OCR→caption логику, что и картинки. `extract_text(path, settings)` остаётся
единой точкой входа — вся остальная часть пайплайна (`kf/ingest.py`) не меняет
свою структуру, только получает доступ к `Settings` и обрабатывает ошибки на
уровне отдельного файла.

**Tech Stack:** Python 3.12, `pytesseract` + системный Tesseract OCR, `Pillow`,
`faster-whisper`, системный `ffmpeg`, OpenRouter (HTTP через `httpx`, уже используется
в `kf/llm.py`). Тесты — `pytest`, реальные интеграционные вызовы без моков там, где
это бесплатно (Tesseract/ffmpeg/faster-whisper), monkeypatch — только для
изоляции логики диспетчеризации в `kf/extract.py` от уже отдельно протестированных
инструментов.

## Global Constraints

- Python ≥3.12, `pythonpath = ["."]` (см. `pyproject.toml`, `[tool.pytest.ini_options]`) — тесты запускаются из корня `Цифровой мозг (digital-brain)/`.
- Зависимости ставятся через `uv sync` после правки `pyproject.toml`.
- Кэш моделей — `settings.model_cache_dir` (уже существующая настройка, используется `fastembed`, переиспользуется и `faster-whisper`).
- Вызовы к OpenRouter (`kf/vision_caption.py::caption_image`) **не покрываются автотестами** — по прецеденту `kf/llm.py::call_llm`, сеть и деньги. Проверяется вручную живым прогоном.
- Каждая задача заканчивается зелёным прогоном `pytest` для затронутых файлов и коммитом в формате `Digital brain | что сделано | версия V X.Y.Z` (см. `CLAUDE.md`/историю коммитов проекта). Версии продолжаются с `V 1.4.0` (спека) — конкретный номер уточняется в шаге коммита каждой задачи.

---

### Task 1: Настройки для OCR/vision/video в конфиге

**Files:**
- Modify: `kf/config.py` (весь файл, ~44 строки)
- Modify: `tests/test_config.py`
- Modify: `.env.example`

**Interfaces:**
- Produces: `Settings.ocr_languages: str`, `Settings.image_caption_threshold_chars: int`, `Settings.vision_model: str`, `Settings.video_frame_interval_seconds: int`, `Settings.whisper_model_size: str` — используются всеми последующими задачами.

- [ ] **Step 1: Написать падающий тест на дефолты новых настроек**

Добавить в `tests/test_config.py`:

```python
def test_defaults_image_and_video_settings(monkeypatch):
    monkeypatch.setenv("POSTGRES_USER", "u")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p")
    monkeypatch.setenv("POSTGRES_DB", "d")
    monkeypatch.setenv("MINIO_ROOT_USER", "m")
    monkeypatch.setenv("MINIO_ROOT_PASSWORD", "s")
    monkeypatch.delenv("OCR_LANGUAGES", raising=False)
    monkeypatch.delenv("IMAGE_CAPTION_THRESHOLD_CHARS", raising=False)
    monkeypatch.delenv("VISION_MODEL", raising=False)
    monkeypatch.delenv("VIDEO_FRAME_INTERVAL_SECONDS", raising=False)
    monkeypatch.delenv("WHISPER_MODEL_SIZE", raising=False)

    settings = load_settings()

    assert settings.ocr_languages == "rus+eng"
    assert settings.image_caption_threshold_chars == 20
    assert settings.vision_model == "google/gemini-2.5-flash"
    assert settings.video_frame_interval_seconds == 15
    assert settings.whisper_model_size == "small"
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `cd "Цифровой мозг (digital-brain)" && uv run pytest tests/test_config.py::test_defaults_image_and_video_settings -v`
Expected: FAIL — `TypeError: Settings.__init__() got an unexpected keyword argument` или `AttributeError` (поля ещё не существуют).

- [ ] **Step 3: Добавить поля в `Settings` и `load_settings`**

Заменить содержимое `kf/config.py` целиком на:

```python
import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class Settings:
    postgres_host: str
    postgres_port: int
    postgres_user: str
    postgres_password: str
    postgres_db: str
    qdrant_url: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    data_root: str
    model_cache_dir: str
    embedding_model: str
    openrouter_api_key: str
    llm_model: str
    ocr_languages: str
    image_caption_threshold_chars: int
    vision_model: str
    video_frame_interval_seconds: int
    whisper_model_size: str


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        postgres_host=os.environ.get("POSTGRES_HOST", "localhost"),
        postgres_port=int(os.environ.get("POSTGRES_PORT", "5432")),
        postgres_user=os.environ["POSTGRES_USER"],
        postgres_password=os.environ["POSTGRES_PASSWORD"],
        postgres_db=os.environ["POSTGRES_DB"],
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        minio_endpoint=os.environ.get("MINIO_ENDPOINT", "localhost:9000"),
        minio_access_key=os.environ["MINIO_ROOT_USER"],
        minio_secret_key=os.environ["MINIO_ROOT_PASSWORD"],
        data_root=os.environ.get("DATA_ROOT", "./data"),
        model_cache_dir=os.environ.get("MODEL_CACHE_DIR", "./data/model-cache"),
        embedding_model=os.environ.get(
            "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        llm_model=os.environ.get("LLM_MODEL", "deepseek/deepseek-v4-flash"),
        ocr_languages=os.environ.get("OCR_LANGUAGES", "rus+eng"),
        image_caption_threshold_chars=int(os.environ.get("IMAGE_CAPTION_THRESHOLD_CHARS", "20")),
        vision_model=os.environ.get("VISION_MODEL", "google/gemini-2.5-flash"),
        video_frame_interval_seconds=int(os.environ.get("VIDEO_FRAME_INTERVAL_SECONDS", "15")),
        whisper_model_size=os.environ.get("WHISPER_MODEL_SIZE", "small"),
    )
```

- [ ] **Step 4: Убедиться, что тест проходит**

Run: `uv run pytest tests/test_config.py -v`
Expected: все тесты в файле PASS (включая новый и три существующих).

- [ ] **Step 5: Задокументировать новые переменные в `.env.example`**

В файле `.env.example` после блока `# --- Эмбеддинги ---` (после строки `MODEL_CACHE_DIR=${DATA_ROOT}/model-cache`) добавить:

```
# --- OCR / vision / видео (картинки и видео в kf.py ingest) ---
# Требует установленных системных Tesseract OCR (+ языковой пакет rus) и ffmpeg — см. ПЕРЕД-СТАРТОМ.md.
OCR_LANGUAGES=rus+eng
IMAGE_CAPTION_THRESHOLD_CHARS=20
VISION_MODEL=google/gemini-2.5-flash
VIDEO_FRAME_INTERVAL_SECONDS=15
WHISPER_MODEL_SIZE=small
```

- [ ] **Step 6: Коммит**

```bash
git add "Цифровой мозг (digital-brain)/kf/config.py" "Цифровой мозг (digital-brain)/tests/test_config.py" "Цифровой мозг (digital-brain)/.env.example"
git commit -m "Digital brain | Настройки OCR/vision/видео в kf/config.py | V 1.4.1"
```

---

### Task 2: Картинки и видео в scope индексации

**Files:**
- Modify: `kf/scope.py`
- Modify: `tests/test_scope.py`
- Modify: `tests/test_ingest.py` (fallout: старый тест использовал `.mp4` как пример исключённого файла)

**Interfaces:**
- Produces: `should_index()` теперь возвращает `True` для `.png .jpg .jpeg .webp .mp4 .mov .mkv .webm`.

- [ ] **Step 1: Написать падающие тесты на новые расширения**

В `tests/test_scope.py` заменить существующий `test_excludes_video` на:

```python
def test_includes_image():
    assert should_index(Path("photos/shot.png")) is True


def test_includes_video():
    assert should_index(Path("clips/intro.mp4")) is True
```

(Удалить старый `test_excludes_video` целиком — он проверял устаревшее поведение.)

- [ ] **Step 2: Убедиться, что новые тесты падают**

Run: `uv run pytest tests/test_scope.py -v`
Expected: `test_includes_image` и `test_includes_video` — FAIL (расширения ещё не в `INCLUDED_EXTENSIONS`).

- [ ] **Step 3: Добавить расширения в `kf/scope.py`**

В `kf/scope.py` заменить:

```python
INCLUDED_EXTENSIONS = {
    ".md",
    ".txt",
    ".pdf",
    ".docx",
    ".csv",
    ".html",
}
```

на:

```python
INCLUDED_EXTENSIONS = {
    ".md",
    ".txt",
    ".pdf",
    ".docx",
    ".csv",
    ".html",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
}
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `uv run pytest tests/test_scope.py -v`
Expected: все PASS.

- [ ] **Step 5: Починить fallout в `tests/test_ingest.py`**

Тест `test_ignores_excluded_files` использовал `.mp4` как пример типа, который
должен игнорироваться — теперь это неверно (видео входит в scope). Заменить его
содержимое на использование действительно исключённого типа:

```python
def test_ignores_excluded_files(tmp_path, deps):
    (tmp_path / "note1.md").write_text("Годная заметка.", encoding="utf-8")
    (tmp_path / "archive.zip").write_bytes(b"not a real archive")

    stats = ingest_directory(tmp_path, deps)

    assert stats.files_scanned == 1
    assert stats.files_ingested == 1
```

(Смысл теста — «файлы вне `INCLUDED_EXTENSIONS` игнорируются», не про видео
конкретно; `.zip` по-прежнему не входит в scope и после этой задачи.)

- [ ] **Step 6: Прогнать полный набор тестов `scope`+`ingest`, убедиться, что ничего не сломано**

Run: `uv run pytest tests/test_scope.py tests/test_ingest.py -v`
Expected: все PASS. (Тесты `test_ingest.py`, требующие живых Qdrant/Postgres/MinIO,
должны быть подняты — `docker ps --filter name=kf-` — иначе они упадут по
несвязанной причине, не по этой задаче.)

- [ ] **Step 7: Коммит**

```bash
git add "Цифровой мозг (digital-brain)/kf/scope.py" "Цифровой мозг (digital-brain)/tests/test_scope.py" "Цифровой мозг (digital-brain)/tests/test_ingest.py"
git commit -m "Digital brain | Картинки и видео в scope индексации kf.py | V 1.4.2"
```

---

### Task 3: OCR-модуль (Tesseract)

**Files:**
- Create: `kf/ocr.py`
- Create: `tests/test_ocr.py`
- Modify: `pyproject.toml` (добавить `pytesseract`, `Pillow` в `dependencies`)

**Interfaces:**
- Produces: `extract_text_from_image(path: Path, languages: str = "rus+eng") -> str`
- Consumes: ничего из предыдущих задач напрямую (использует `settings.ocr_languages` только через вызывающую сторону в Task 7).

**Предварительное условие:** установлен системный Tesseract OCR (Windows-инсталлятор
+ языковой пакет `rus`), доступен в `PATH` как `tesseract`. Если не установлен —
шаг 4 упадёт с понятной ошибкой `pytesseract.TesseractNotFoundError`; установить
перед продолжением (см. также Task 9 — обновление `ПЕРЕД-СТАРТОМ.md`).

- [ ] **Step 1: Добавить зависимости**

В `pyproject.toml`, в блок `dependencies = [...]`, добавить (сохраняя алфавитный
порядок существующих строк):

```
    "pillow>=11.0.0",
    "pytesseract>=0.3.13",
```

Run: `cd "Цифровой мозг (digital-brain)" && uv sync`
Expected: зависимости установлены без ошибок.

- [ ] **Step 2: Написать падающий тест**

Создать `tests/test_ocr.py`:

```python
from PIL import Image, ImageDraw, ImageFont

from kf.ocr import extract_text_from_image


def test_recognizes_text_in_generated_image(tmp_path):
    img = Image.new("RGB", (400, 120), color="white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 40)
    draw.text((10, 30), "HELLO WORLD", fill="black", font=font)
    f = tmp_path / "screenshot.png"
    img.save(f)

    text = extract_text_from_image(f, languages="eng")

    assert "HELLO" in text.upper()
```

- [ ] **Step 3: Убедиться, что тест падает**

Run: `uv run pytest tests/test_ocr.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kf.ocr'`.

- [ ] **Step 4: Реализовать `kf/ocr.py`**

```python
from pathlib import Path

import pytesseract
from PIL import Image


def extract_text_from_image(path: Path, languages: str = "rus+eng") -> str:
    image = Image.open(path)
    return pytesseract.image_to_string(image, lang=languages).strip()
```

- [ ] **Step 5: Убедиться, что тест проходит**

Run: `uv run pytest tests/test_ocr.py -v`
Expected: PASS. Если падает с `TesseractNotFoundError` — Tesseract не установлен
или не в `PATH`, установить перед продолжением плана.

- [ ] **Step 6: Коммит**

```bash
git add "Цифровой мозг (digital-brain)/kf/ocr.py" "Цифровой мозг (digital-brain)/tests/test_ocr.py" "Цифровой мозг (digital-brain)/pyproject.toml" "Цифровой мозг (digital-brain)/uv.lock"
git commit -m "Digital brain | Локальный OCR-модуль (Tesseract) для kf.py | V 1.4.3"
```

---

### Task 4: Vision-caption модуль (OpenRouter)

**Files:**
- Create: `kf/vision_caption.py`
- Create: `tests/test_vision_caption.py`

**Interfaces:**
- Consumes: `kf.config.Settings` (`openrouter_api_key`, `vision_model` — из Task 1).
- Produces: `build_vision_messages(image_path) -> list[dict]` (тестируется), `caption_image(settings: Settings, image_path) -> str` (сетевой вызов, не тестируется — см. Global Constraints).

- [ ] **Step 1: Написать падающий тест на построение сообщения**

Создать `tests/test_vision_caption.py`:

```python
from kf.vision_caption import build_vision_messages


def test_builds_message_with_base64_image_and_png_mime(tmp_path):
    f = tmp_path / "photo.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\nfakepngbytes")

    messages = build_vision_messages(f)

    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_uses_jpeg_mime_for_jpg_extension(tmp_path):
    f = tmp_path / "photo.jpg"
    f.write_bytes(b"fakejpegbytes")

    messages = build_vision_messages(f)

    assert messages[0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_vision_caption.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kf.vision_caption'`.

- [ ] **Step 3: Реализовать `kf/vision_caption.py`**

```python
import base64
from pathlib import Path

import httpx

from kf.config import Settings

_MIME_BY_SUFFIX = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}

_DEFAULT_PROMPT = "Опиши, что изображено на картинке, подробно и по-русски."


def build_vision_messages(image_path: Path, prompt: str = _DEFAULT_PROMPT) -> list[dict]:
    image_bytes = Path(image_path).read_bytes()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    suffix = Path(image_path).suffix.lower().lstrip(".")
    mime = _MIME_BY_SUFFIX.get(suffix, "image/png")
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }
    ]


def caption_image(settings: Settings, image_path: Path) -> str:
    messages = build_vision_messages(image_path)
    response = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
        json={"model": settings.vision_model, "messages": messages},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
```

- [ ] **Step 4: Убедиться, что тест проходит**

Run: `uv run pytest tests/test_vision_caption.py -v`
Expected: PASS.

- [ ] **Step 5: Коммит**

```bash
git add "Цифровой мозг (digital-brain)/kf/vision_caption.py" "Цифровой мозг (digital-brain)/tests/test_vision_caption.py"
git commit -m "Digital brain | Vision-caption модуль через OpenRouter для kf.py | V 1.4.4"
```

---

### Task 5: Видео-модуль (ffmpeg: звук + кадры)

**Files:**
- Create: `kf/video.py`
- Create: `tests/test_video.py`

**Interfaces:**
- Produces: `extract_audio(video_path: Path) -> Path`, `sample_frames(video_path: Path, interval_seconds: int) -> list[Path]`

**Предварительное условие:** `ffmpeg` установлен и доступен в `PATH`. Если нет —
шаг 4 упадёт с `FileNotFoundError`/`subprocess.CalledProcessError`; установить
перед продолжением (см. Task 9).

- [ ] **Step 1: Написать падающие тесты**

Создать `tests/test_video.py`:

```python
import subprocess

from kf.video import extract_audio, sample_frames


def _make_test_video(path, duration=3):
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=320x240:rate=10",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
            "-c:v", "libx264", "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def test_extract_audio_produces_nonempty_wav(tmp_path):
    video = tmp_path / "clip.mp4"
    _make_test_video(video)

    audio_path = extract_audio(video)

    assert audio_path.exists()
    assert audio_path.stat().st_size > 0


def test_sample_frames_produces_expected_frame_count(tmp_path):
    video = tmp_path / "clip.mp4"
    _make_test_video(video, duration=3)

    frames = sample_frames(video, interval_seconds=1)

    assert len(frames) >= 2
    assert all(f.exists() for f in frames)
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `uv run pytest tests/test_video.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kf.video'`.

- [ ] **Step 3: Реализовать `kf/video.py`**

```python
import subprocess
import tempfile
from pathlib import Path


def extract_audio(video_path: Path) -> Path:
    output_path = Path(tempfile.gettempdir()) / f"{Path(video_path).stem}_audio.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def sample_frames(video_path: Path, interval_seconds: int) -> list[Path]:
    frames_dir = Path(tempfile.mkdtemp(prefix="kf_frames_"))
    pattern = frames_dir / "frame_%04d.png"
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", f"fps=1/{interval_seconds}",
            str(pattern),
        ],
        check=True,
        capture_output=True,
    )
    return sorted(frames_dir.glob("frame_*.png"))
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `uv run pytest tests/test_video.py -v`
Expected: PASS. Если падает из-за отсутствия `ffmpeg` — установить и повторить.

- [ ] **Step 5: Коммит**

```bash
git add "Цифровой мозг (digital-brain)/kf/video.py" "Цифровой мозг (digital-brain)/tests/test_video.py"
git commit -m "Digital brain | Видео-модуль (ffmpeg: звук + кадры) для kf.py | V 1.4.5"
```

---

### Task 6: Транскрипция звука (faster-whisper)

**Files:**
- Create: `kf/transcribe.py`
- Create: `tests/test_transcribe.py`
- Modify: `pyproject.toml` (добавить `faster-whisper` в `dependencies`, `pyttsx3` в `[dependency-groups].dev`)

**Interfaces:**
- Produces: `transcribe_audio(path: Path, model_size: str = "small", cache_dir: str = "./data/model-cache") -> str`

- [ ] **Step 1: Добавить зависимости**

В `pyproject.toml`, в `dependencies = [...]` добавить:
```
    "faster-whisper>=1.1.0",
```
В `[dependency-groups].dev = [...]` добавить:
```
    "pyttsx3>=2.98",
```

Run: `cd "Цифровой мозг (digital-brain)" && uv sync`
Expected: зависимости установлены без ошибок.

- [ ] **Step 2: Написать падающий тест**

Создать `tests/test_transcribe.py`:

```python
import pyttsx3

from kf.transcribe import transcribe_audio


def test_transcribes_synthesized_speech(tmp_path):
    audio_path = tmp_path / "speech.wav"
    engine = pyttsx3.init()
    engine.save_to_file("testing one two three", str(audio_path))
    engine.runAndWait()

    text = transcribe_audio(audio_path, model_size="small", cache_dir=str(tmp_path / "model-cache"))

    assert "test" in text.lower()
```

- [ ] **Step 3: Убедиться, что тест падает**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kf.transcribe'`.

- [ ] **Step 4: Реализовать `kf/transcribe.py`**

```python
from pathlib import Path

from faster_whisper import WhisperModel


def transcribe_audio(path: Path, model_size: str = "small", cache_dir: str = "./data/model-cache") -> str:
    model = WhisperModel(model_size, device="cpu", compute_type="int8", download_root=cache_dir)
    segments, _ = model.transcribe(str(path))
    return " ".join(segment.text.strip() for segment in segments).strip()
```

- [ ] **Step 5: Убедиться, что тест проходит**

Run: `uv run pytest tests/test_transcribe.py -v`
Expected: PASS. Первый запуск скачает модель `small` в `cache_dir` (нужен интернет
один раз, дальше — из кэша). Если `pyttsx3.init()` падает (нет установленного
голоса SAPI5) — на используемой машине это не ожидается (Windows, голос по
умолчанию есть), но при проблеме — установить любой англоязычный голос Windows.

- [ ] **Step 6: Коммит**

```bash
git add "Цифровой мозг (digital-brain)/kf/transcribe.py" "Цифровой мозг (digital-brain)/tests/test_transcribe.py" "Цифровой мозг (digital-brain)/pyproject.toml" "Цифровой мозг (digital-brain)/uv.lock"
git commit -m "Digital brain | Транскрипция звука (faster-whisper) для kf.py | V 1.4.6"
```

---

### Task 7: Диспетчеризация в `extract_text` (картинки + видео)

**Files:**
- Modify: `kf/extract.py` (весь файл)
- Modify: `tests/test_extract.py`

**Interfaces:**
- Consumes: `kf.ocr.extract_text_from_image`, `kf.vision_caption.caption_image`, `kf.video.extract_audio`, `kf.video.sample_frames`, `kf.transcribe.transcribe_audio`, `kf.config.Settings` (все — из Task 1, 3, 4, 5, 6).
- Produces: `extract_text(path: Path, settings: Settings) -> str` — **сигнатура меняется** (было `extract_text(path)`), потребители — `kf/ingest.py` (Task 8).

- [ ] **Step 1: Добавить общий хелпер настроек и обновить существующие тесты под новую сигнатуру**

В начало `tests/test_extract.py` добавить хелпер и обновить 4 существующих теста,
передав `settings` вторым аргументом. Полное новое содержимое файла:

```python
from docx import Document
from fpdf import FPDF

from kf.config import Settings
from kf.extract import extract_text


def _dummy_settings(**overrides) -> Settings:
    base = dict(
        postgres_host="localhost", postgres_port=5432, postgres_user="u",
        postgres_password="p", postgres_db="d", qdrant_url="http://localhost:6333",
        minio_endpoint="localhost:9000", minio_access_key="a", minio_secret_key="s",
        data_root="./data", model_cache_dir="./data/model-cache", embedding_model="m",
        openrouter_api_key="k", llm_model="l", ocr_languages="eng",
        image_caption_threshold_chars=20, vision_model="v",
        video_frame_interval_seconds=15, whisper_model_size="small",
    )
    base.update(overrides)
    return Settings(**base)


def test_extracts_plain_markdown(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# Заголовок\n\nТело заметки.", encoding="utf-8")

    assert extract_text(f, _dummy_settings()) == "# Заголовок\n\nТело заметки."


def test_extracts_csv_as_text(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b\n1,2", encoding="utf-8")

    assert extract_text(f, _dummy_settings()) == "a,b\n1,2"


def test_extracts_docx_paragraphs(tmp_path):
    f = tmp_path / "report.docx"
    doc = Document()
    doc.add_paragraph("Первый абзац.")
    doc.add_paragraph("Второй абзац.")
    doc.save(f)

    text = extract_text(f, _dummy_settings())

    assert "Первый абзац." in text
    assert "Второй абзац." in text


def test_extracts_pdf_text(tmp_path):
    f = tmp_path / "doc.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(text="Hello from test PDF")
    pdf.output(str(f))

    text = extract_text(f, _dummy_settings())

    assert "Hello from test PDF" in text


def test_dispatches_image_to_ocr(tmp_path, monkeypatch):
    f = tmp_path / "shot.png"
    f.write_bytes(b"fakepng")
    monkeypatch.setattr("kf.extract.extract_text_from_image", lambda path, languages: "OCR TEXT HERE")

    text = extract_text(f, _dummy_settings(image_caption_threshold_chars=5))

    assert text == "OCR TEXT HERE"


def test_falls_back_to_caption_when_ocr_text_is_short(tmp_path, monkeypatch):
    f = tmp_path / "photo.jpg"
    f.write_bytes(b"fakejpg")
    monkeypatch.setattr("kf.extract.extract_text_from_image", lambda path, languages: "")
    monkeypatch.setattr("kf.extract.caption_image", lambda settings, path: "Фото заката над морем")

    text = extract_text(f, _dummy_settings(image_caption_threshold_chars=20))

    assert "Фото заката над морем" in text


def test_dispatches_video_to_transcript_and_frames(tmp_path, monkeypatch):
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"fakevideo")
    fake_audio = tmp_path / "audio.wav"
    fake_frame = tmp_path / "frame_0000.png"
    fake_frame.write_bytes(b"fakeframe")

    monkeypatch.setattr("kf.extract.extract_audio", lambda path: fake_audio)
    monkeypatch.setattr(
        "kf.extract.transcribe_audio", lambda path, model_size, cache_dir: "Привет мир"
    )
    monkeypatch.setattr(
        "kf.extract.sample_frames", lambda path, interval_seconds: [fake_frame]
    )
    monkeypatch.setattr(
        "kf.extract.extract_text_from_image", lambda path, languages: "текст на кадре"
    )

    text = extract_text(f, _dummy_settings(image_caption_threshold_chars=5))

    assert "[Транскрипт]" in text
    assert "Привет мир" in text
    assert "[Кадр 00:00]" in text
    assert "текст на кадре" in text
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `uv run pytest tests/test_extract.py -v`
Expected: 4 старых теста FAIL (`TypeError: extract_text() missing 1 required
positional argument: 'settings'`), 3 новых теста FAIL (нет диспетчеризации).

- [ ] **Step 3: Реализовать диспетчеризацию в `kf/extract.py`**

Заменить содержимое `kf/extract.py` целиком на:

```python
from pathlib import Path

from docx import Document
from pypdf import PdfReader

from kf.config import Settings
from kf.ocr import extract_text_from_image
from kf.transcribe import transcribe_audio
from kf.video import extract_audio, sample_frames
from kf.vision_caption import caption_image

PLAIN_TEXT_EXTENSIONS = {".md", ".txt", ".csv", ".html"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}


def _extract_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text)


def _extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_image(path: Path, settings: Settings) -> str:
    ocr_text = extract_text_from_image(path, languages=settings.ocr_languages)
    if len(ocr_text.strip()) < settings.image_caption_threshold_chars:
        caption = caption_image(settings, path)
        return f"{ocr_text}\n\n[Описание изображения]\n{caption}".strip()
    return ocr_text


def _extract_video(path: Path, settings: Settings) -> str:
    audio_path = extract_audio(path)
    transcript = transcribe_audio(
        audio_path, model_size=settings.whisper_model_size, cache_dir=settings.model_cache_dir
    )

    frames = sample_frames(path, interval_seconds=settings.video_frame_interval_seconds)
    frame_blocks = []
    for i, frame_path in enumerate(frames):
        timestamp_seconds = i * settings.video_frame_interval_seconds
        minutes, seconds = divmod(timestamp_seconds, 60)
        frame_text = _extract_image(frame_path, settings)
        frame_blocks.append(f"[Кадр {minutes:02d}:{seconds:02d}]\n{frame_text}")

    parts = [f"[Транскрипт]\n{transcript}"] + frame_blocks
    return "\n\n".join(parts).strip()


def extract_text(path: Path, settings: Settings) -> str:
    suffix = path.suffix.lower()
    if suffix in PLAIN_TEXT_EXTENSIONS:
        return path.read_text(encoding="utf-8")
    if suffix == ".docx":
        return _extract_docx(path)
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix in IMAGE_EXTENSIONS:
        return _extract_image(path, settings)
    if suffix in VIDEO_EXTENSIONS:
        return _extract_video(path, settings)
    raise ValueError(f"Unsupported file type: {suffix}")
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `uv run pytest tests/test_extract.py -v`
Expected: все 7 тестов PASS.

- [ ] **Step 5: Коммит**

```bash
git add "Цифровой мозг (digital-brain)/kf/extract.py" "Цифровой мозг (digital-brain)/tests/test_extract.py"
git commit -m "Digital brain | Диспетчеризация extract_text для картинок и видео | V 1.4.7"
```

---

### Task 8: Прокинуть settings и обработку ошибок в `ingest_directory`

**Files:**
- Modify: `kf/ingest.py` (весь файл)
- Modify: `kf/cli.py:14-24,44-51`
- Modify: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `extract_text(path, settings)` (Task 7).
- Produces: `IngestDeps.settings: Settings` (новое обязательное поле), `IngestStats.files_failed: int` (новое поле).

- [ ] **Step 1: Обновить фикстуру `deps` и добавить падающий тест на обработку ошибок**

В `tests/test_ingest.py`, в фикстуре `deps`, добавить `settings=settings` в
конструктор `IngestDeps` (после `collection=COLLECTION,`):

```python
    d = IngestDeps(
        pg_conn=pg_conn,
        qdrant_client=qdrant_client,
        minio_client=minio_client,
        embedder=embedder,
        collection=COLLECTION,
        settings=settings,
    )
```

Добавить новый тест в конец файла:

```python
def test_failed_file_does_not_abort_whole_run(tmp_path, deps):
    (tmp_path / "note1.md").write_text("Годная заметка.", encoding="utf-8")
    (tmp_path / "broken.png").write_bytes(b"not a real png")

    stats = ingest_directory(tmp_path, deps)

    assert stats.files_scanned == 2
    assert stats.files_ingested == 1
    assert stats.files_failed == 1
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: все тесты FAIL с `TypeError: IngestDeps.__init__() got an unexpected
keyword argument 'settings'` (фикстура ломает вообще все тесты файла — ожидаемо,
так как `settings` пока не существует в `IngestDeps`).

- [ ] **Step 3: Реализовать изменения в `kf/ingest.py`**

Заменить содержимое `kf/ingest.py` целиком на:

```python
import uuid
from dataclasses import dataclass
from pathlib import Path

from kf.chunking import chunk_text
from kf.config import Settings
from kf.embeddings import embed
from kf.extract import extract_text
from kf.hashing import sha256_of_file
from kf.scope import should_index
from kf.store.minio_store import upload_file
from kf.store.postgres import needs_ingest, record_ingested
from kf.store.qdrant_store import upsert_chunks

_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


@dataclass
class IngestDeps:
    pg_conn: object
    qdrant_client: object
    minio_client: object
    embedder: object
    collection: str
    settings: Settings
    max_chars: int = 1500
    overlap: int = 150


@dataclass
class IngestStats:
    files_scanned: int = 0
    files_ingested: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    chunks_written: int = 0


def _point_id(path: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{path}:{chunk_index}"))


def ingest_directory(source_dir: Path, deps: IngestDeps) -> IngestStats:
    stats = IngestStats()

    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or not should_index(path):
            continue

        rel_key = path.relative_to(source_dir).as_posix()
        stats.files_scanned += 1

        file_hash = sha256_of_file(path)
        if not needs_ingest(deps.pg_conn, rel_key, file_hash):
            stats.files_skipped += 1
            continue

        try:
            text = extract_text(path, deps.settings)
        except Exception as exc:
            print(f"[ingest] пропускаю {rel_key}: {exc}")
            stats.files_failed += 1
            continue

        chunks = chunk_text(text, max_chars=deps.max_chars, overlap=deps.overlap)
        if chunks:
            vectors = embed(deps.embedder, chunks)
            points = [
                {
                    "id": _point_id(rel_key, i),
                    "vector": vectors[i],
                    "payload": {"path": rel_key, "chunk_index": i, "text": chunks[i]},
                }
                for i in range(len(chunks))
            ]
            upsert_chunks(deps.qdrant_client, deps.collection, points)
            stats.chunks_written += len(points)

        upload_file(deps.minio_client, path, rel_key)
        record_ingested(deps.pg_conn, rel_key, file_hash)
        stats.files_ingested += 1

    return stats
```

- [ ] **Step 4: Обновить `kf/cli.py`**

В `kf/cli.py` заменить функцию `_build_ingest_deps`:

```python
def _build_ingest_deps(settings) -> IngestDeps:
    session = open_session()
    minio_client = get_minio_client(settings)
    ensure_bucket(minio_client)
    return IngestDeps(
        pg_conn=session.pg_conn,
        qdrant_client=session.qdrant_client,
        minio_client=minio_client,
        embedder=session.embedder,
        collection=COLLECTION,
        settings=settings,
    )
```

И в команде `ingest` заменить финальный `click.echo(...)`:

```python
    click.echo(
        f"Готово. просканировано: {stats.files_scanned}, "
        f"проиндексировано: {stats.files_ingested}, "
        f"пропущено (без изменений): {stats.files_skipped}, "
        f"ошибок: {stats.files_failed}, "
        f"чанков записано: {stats.chunks_written}"
    )
```

- [ ] **Step 5: Убедиться, что тесты проходят**

Run: `uv run pytest tests/test_ingest.py tests/test_cli.py -v`
Expected: все PASS (стек `docker ps --filter name=kf-` должен быть поднят и
healthy — эти тесты интеграционные, против реальных Qdrant/Postgres/MinIO).

- [ ] **Step 6: Прогнать весь набор тестов проекта**

Run: `uv run pytest -v`
Expected: все тесты PASS (было 51, теперь больше — новые модули + новые кейсы).

- [ ] **Step 7: Коммит**

```bash
git add "Цифровой мозг (digital-brain)/kf/ingest.py" "Цифровой мозг (digital-brain)/kf/cli.py" "Цифровой мозг (digital-brain)/tests/test_ingest.py"
git commit -m "Digital brain | settings + обработка ошибок на уровне файла в ingest_directory | V 1.4.8"
```

---

### Task 9: Документация — ручная установка Tesseract/ffmpeg

**Files:**
- Modify: `Файлы настройки проекта/Архитектура проекта. Настройка и Правила/ПЕРЕД-СТАРТОМ.md:9-13`

**Interfaces:** нет (документация, без кода).

- [ ] **Step 1: Добавить пункты в раздел «1. ЧТО ТЕБЕ ПОНАДОБИТСЯ»**

В файле `Файлы настройки проекта/Архитектура проекта. Настройка и Правила/ПЕРЕД-СТАРТОМ.md`
заменить:

```
## 1. ЧТО ТЕБЕ ПОНАДОБИТСЯ
- **Docker** (Desktop/OrbStack/Engine) + `docker compose`.
- **Отдельный диск или папка с запасом** ≥ 50 GB под данные (внешний SSD / не системный раздел).
- **Системный диск** ≥ 10 GB свободно под образы Docker.
- Вариант LLM (выбери один в п.4): локальный (Ollama) **или** через OpenRouter-ключ.
```

на:

```
## 1. ЧТО ТЕБЕ ПОНАДОБИТСЯ
- **Docker** (Desktop/OrbStack/Engine) + `docker compose`.
- **Отдельный диск или папка с запасом** ≥ 50 GB под данные (внешний SSD / не системный раздел).
- **Системный диск** ≥ 10 GB свободно под образы Docker.
- Вариант LLM (выбери один в п.4): локальный (Ollama) **или** через OpenRouter-ключ.
- **Tesseract OCR** (+ языковой пакет `rus`) — для распознавания текста на
  картинках/скриншотах в `kf.py ingest`. Windows: инсталлятор с
  github.com/UB-Mannheim/tesseract/wiki, при установке отметить галку `rus` в
  выборе языковых пакетов. Проверка: `tesseract --version` в терминале.
- **ffmpeg** в `PATH` — для извлечения звука и кадров из видео в `kf.py ingest`.
  Windows: скачать сборку с gyan.dev/ffmpeg/builds, добавить папку `bin/` в
  `PATH`. Проверка: `ffmpeg -version` в терминале.
```

- [ ] **Step 2: Проверить, что файл рендерится корректно**

Run: откройте файл и визуально сверьте, что новые пункты списка не сломали
markdown-форматирование (отступы `-`/дефисов, отсутствие лишних пустых строк).

- [ ] **Step 3: Коммит**

```bash
git add "Файлы настройки проекта/Архитектура проекта. Настройка и Правила/ПЕРЕД-СТАРТОМ.md"
git commit -m "Digital brain | Инструкция по установке Tesseract/ffmpeg в ПЕРЕД-СТАРТОМ.md | V 1.4.9"
```

---

## После завершения плана

Ручная проверка (не покрыта автотестами по Global Constraints):
1. Положить реальный скриншот с текстом в тестовую папку, прогнать
   `kf.py ingest --source <папка>`, проверить `kf.py search "<кусок текста со
   скриншота>"` находит его.
2. Положить короткое реальное видео (с речью), прогнать ingest, проверить
   `kf.py ask` отвечает на вопрос по содержанию видео (речь) и `kf.py search`
   находит текст, распознанный на кадрах (если на видео был текст).
3. Убедиться, что `IMAGE_CAPTION_THRESHOLD_CHARS`/`VISION_MODEL` действительно
   включают fallback на OpenRouter — подсунуть фото без текста, проверить, что
   в итоговом тексте есть блок `[Описание изображения]`.
