# Расширение форматов (ссылки, аудио, Excel) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Добавить в `kf.py` поддержку ссылок (статьи и YouTube-видео через новую команду
`kf.py ingest-url`), отдельных аудиофайлов/голосовых записей, и Excel-таблиц (.xlsx) — три
формата, которые пользователь ожидает от регламента автодобавления знаний, но которые сейчас
не работают.

**Architecture:** Новый модуль `kf/web_extract.py` (httpx + trafilatura для статей,
youtube-transcript-api → yt-dlp+Whisper для видео, тот же паттерн "чистая функция + тонкий
сетевой вызов", что уже применён в `kf/graph.py` и `kf/embeddings.py`). Новая CLI-команда
`kf.py ingest-url` переиспользует существующие `_build_ingest_deps`/`ingest_directory` без
дублирования логики. Аудио и Excel — новые ветки диспетчеризации в уже существующем
`kf/extract.py`, по тому же паттерну, что уже есть для docx/pdf/изображений/видео.

**Tech Stack:** Python 3.12, `trafilatura` (извлечение текста статей), `yt-dlp` (скачивание
аудио с видеохостингов), `youtube-transcript-api` (готовые субтитры YouTube), `openpyxl`
(чтение .xlsx), httpx (уже используется), pytest, click.

## Global Constraints

- `trafilatura` — основной, автоматический путь для статей. Firecrawl НЕ встраивается в
  `kf.py` — он доступен только в чате ассистента (MCP), а `kf.py` — самостоятельный офлайн-
  скрипт. Эскалация на Firecrawl остаётся ручным шагом ассистента вне `kf.py`.
- Для видео: сначала `youtube-transcript-api` (готовые субтитры, дёшево), при отсутствии
  субтитров или ошибке — `yt-dlp` (скачивание аудио) + уже существующий `transcribe_audio`.
  Работает только для YouTube (`youtube.com`/`youtu.be`) — прочие хостинги сразу идут через
  скачивание.
- Если извлечённый текст статьи короче порога (эвристика "подозрительно скудный текст",
  аналогично уже существующему `image_caption_threshold_chars`) — команда не проваливается
  молча, а печатает явное предупреждение и всё равно сохраняет то, что получилось.
- Новые расширения (`.mp3`, `.wav`, `.ogg`, `.m4a`, `.xlsx`) должны быть добавлены и в
  `kf/extract.py` (как их читать), И в `kf/scope.py::INCLUDED_EXTENSIONS` (иначе
  `ingest_directory` их не увидит при сканировании папки вообще) — оба слоя обязательны.
- Сетевые вызовы (`extract_article`, `extract_youtube_transcript`,
  `extract_video_via_download`) не покрываются юнит-тестами — тот же принцип, что уже принят
  для LLM-вызова в `kf/graph.py` и HTTP-вызова OpenRouter в `kf/embeddings.py`; проверяются
  вручную на реальных данных при финальной верификации фичи.
- Вне рамок этого плана: встраивание Firecrawl/Apify в `kf.py`, поддержка видеохостингов
  кроме YouTube для пути "готовые субтитры", перенос Surya OCR/whisperx/PySceneDetect из
  проекта "Marketing agency Project".

---

### Task 1: Модуль `kf/web_extract.py` — извлечение из ссылок

**Files:**
- Create: `knowledge-factory/kf/web_extract.py`
- Test: `knowledge-factory/tests/test_web_extract.py`
- Modify: `knowledge-factory/pyproject.toml` (добавить `trafilatura`, `yt-dlp`,
  `youtube-transcript-api`)

**Interfaces:**
- Consumes: `kf.config.Settings` (существующий тип), `kf.transcribe.transcribe_audio`
  (существующая функция).
- Produces: `is_youtube_url(url: str) -> bool`, `derive_filename(title: str | None) -> str`,
  `extract_article(url: str) -> tuple[str, str | None]` (текст, заголовок-или-None),
  `extract_youtube_transcript(url: str) -> str | None` (None — субтитров нет, не ошибка),
  `extract_video_via_download(url: str, settings: Settings) -> str`,
  `extract_from_url(url: str, settings: Settings) -> tuple[str, str | None, bool]` (текст,
  заголовок-или-None, признак низкого качества). Используются в Task 2 (`kf/cli.py`).

- [ ] **Step 1: Add dependencies**

Run: `cd "knowledge-factory" && uv add trafilatura yt-dlp youtube-transcript-api`

Проверить, что `pyproject.toml` теперь содержит все три пакета в `dependencies`, и что
`uv.lock` обновился.

- [ ] **Step 2: Write the failing tests**

Создать `knowledge-factory/tests/test_web_extract.py`:

```python
import pytest

from kf.web_extract import derive_filename, is_youtube_url


def test_is_youtube_url_detects_standard_domain():
    assert is_youtube_url("https://www.youtube.com/watch?v=abc123") is True


def test_is_youtube_url_detects_short_domain():
    assert is_youtube_url("https://youtu.be/abc123") is True


def test_is_youtube_url_rejects_other_domains():
    assert is_youtube_url("https://example.com/article") is False


def test_derive_filename_slugifies_title():
    assert derive_filename("Как настроить Docker: полное руководство!") == "Как-настроить-Docker-полное-руководство"


def test_derive_filename_truncates_long_title():
    long_title = "Слово " * 30
    result = derive_filename(long_title)

    assert len(result) <= 80


def test_derive_filename_falls_back_to_timestamp_when_no_title():
    result = derive_filename(None)

    assert result.startswith("url-")


def test_derive_filename_falls_back_to_timestamp_when_title_has_no_word_chars():
    result = derive_filename("!!!???")

    assert result.startswith("url-")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_web_extract.py -v`
Expected: FAIL с `ModuleNotFoundError: No module named 'kf.web_extract'`.

- [ ] **Step 4: Implement**

Создать `knowledge-factory/kf/web_extract.py`:

```python
import re
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import trafilatura
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

from kf.config import Settings
from kf.transcribe import transcribe_audio

LOW_QUALITY_THRESHOLD_CHARS = 200


def is_youtube_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return "youtube.com" in host or "youtu.be" in host


def derive_filename(title: str | None) -> str:
    if title:
        slug = re.sub(r"[^\w\s-]", "", title, flags=re.UNICODE).strip()
        slug = re.sub(r"\s+", "-", slug)[:80]
        if slug:
            return slug
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"url-{timestamp}"


def extract_article(url: str) -> tuple[str, str | None]:
    response = httpx.get(url, timeout=30, follow_redirects=True)
    response.raise_for_status()
    downloaded = response.text
    text = trafilatura.extract(downloaded) or ""
    metadata = trafilatura.extract_metadata(downloaded)
    title = metadata.title if metadata else None
    return text, title


def _youtube_video_id(url: str) -> str | None:
    parsed = urlparse(url)
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/") or None
    query_id = parse_qs(parsed.query).get("v")
    return query_id[0] if query_id else None


def extract_youtube_transcript(url: str) -> str | None:
    video_id = _youtube_video_id(url)
    if not video_id:
        return None
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=["ru", "en"])
    except Exception:
        return None
    return " ".join(snippet.text for snippet in transcript).strip()


def extract_video_via_download(url: str, settings: Settings) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        outtmpl = str(Path(tmpdir) / "audio.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
            "quiet": True,
            "noprogress": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        downloaded = list(Path(tmpdir).glob("audio.*"))
        if not downloaded:
            raise RuntimeError("yt-dlp не смог скачать аудио по этой ссылке")
        return transcribe_audio(downloaded[0], settings.whisper_model_size, settings.model_cache_dir)


def extract_from_url(url: str, settings: Settings) -> tuple[str, str | None, bool]:
    if is_youtube_url(url):
        text = extract_youtube_transcript(url)
        if text is None:
            text = extract_video_via_download(url, settings)
        title = None
    else:
        text, title = extract_article(url)

    is_low_quality = len(text.strip()) < LOW_QUALITY_THRESHOLD_CHARS
    return text, title, is_low_quality
```

Важно: `youtube-transcript-api`'s публичный интерфейс менялся между major-версиями (старые
версии — статический `YouTubeTranscriptApi.get_transcript(video_id, languages=[...])`,
возвращающий список словарей `{"text": ...}`; новые (1.x) — `YouTubeTranscriptApi().fetch(...)`,
возвращающий объект с `.text` на каждом элементе). Код выше рассчитан на новый интерфейс.
Если тесты Task 1 не требуют реального вызова (`extract_youtube_transcript` не покрыт юнит-
тестами по Global Constraints), несовпадение проявится только при ручной проверке на реальных
данных в конце фичи — на этом шаге просто проверить фактическую версию через
`uv run python -c "import youtube_transcript_api; print(youtube_transcript_api.__version__)"`
и при необходимости поправить вызов внутри `extract_youtube_transcript`, сохранив сигнатуру
функции (`url: str -> str | None`) неизменной.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_web_extract.py -v`
Expected: PASS (7/7).

- [ ] **Step 6: Commit**

```bash
git add "knowledge-factory/pyproject.toml" "knowledge-factory/uv.lock" "knowledge-factory/kf/web_extract.py" "knowledge-factory/tests/test_web_extract.py"
git commit -m "Digital brain | Модуль kf/web_extract.py: извлечение текста из ссылок (статьи + YouTube) | V 1.4.51"
```

---

### Task 2: Команда `kf.py ingest-url`

**Files:**
- Modify: `knowledge-factory/kf/cli.py`
- Test: `knowledge-factory/tests/test_cli.py`

**Interfaces:**
- Consumes: `extract_from_url`, `derive_filename` из Task 1 (`kf.web_extract`);
  `_build_ingest_deps`, `ingest_directory` (существующие, уже используются командой `ingest`).
- Produces: CLI-команда `kf.py ingest-url <url> --dest <папка>`.

- [ ] **Step 1: Write the failing tests**

Добавить в `knowledge-factory/tests/test_cli.py`:

```python
def test_ingest_url_saves_file_and_reports_path(tmp_path, monkeypatch):
    settings = load_settings()
    monkeypatch.setattr("kf.cli.load_settings", lambda: settings)
    monkeypatch.setattr("kf.cli.DEFAULT_SOURCE", tmp_path)
    monkeypatch.setattr(
        "kf.cli.extract_from_url",
        lambda url, settings: ("Текст статьи про Docker.", "Статья про Docker", False),
    )

    captured = {}

    def _fake_build_deps(settings):
        captured["called"] = True
        return object()

    def _fake_ingest_directory(source_dir, deps, detect_deletions=True):
        from kf.ingest import IngestStats

        captured["source_dir"] = source_dir
        captured["detect_deletions"] = detect_deletions
        return IngestStats(files_scanned=1, files_ingested=1, chunks_written=2)

    monkeypatch.setattr("kf.cli._build_ingest_deps", _fake_build_deps)
    monkeypatch.setattr("kf.cli.ingest_directory", _fake_ingest_directory)

    runner = CliRunner()
    result = runner.invoke(cli, ["ingest-url", "https://example.com/docker", "--dest", "003 Знания"])

    assert result.exit_code == 0
    saved_file = tmp_path / "003 Знания" / "Статья-про-Docker.md"
    assert saved_file.exists()
    assert saved_file.read_text(encoding="utf-8") == "Текст статьи про Docker."
    assert "003 Знания" in result.output
    assert captured["detect_deletions"] is False


def test_ingest_url_warns_on_low_quality_extraction(tmp_path, monkeypatch):
    settings = load_settings()
    monkeypatch.setattr("kf.cli.load_settings", lambda: settings)
    monkeypatch.setattr("kf.cli.DEFAULT_SOURCE", tmp_path)
    monkeypatch.setattr(
        "kf.cli.extract_from_url", lambda url, settings: ("", None, True)
    )
    monkeypatch.setattr("kf.cli._build_ingest_deps", lambda settings: object())

    def _fake_ingest_directory(source_dir, deps, detect_deletions=True):
        from kf.ingest import IngestStats

        return IngestStats()

    monkeypatch.setattr("kf.cli.ingest_directory", _fake_ingest_directory)

    runner = CliRunner()
    result = runner.invoke(cli, ["ingest-url", "https://example.com/hard-site", "--dest", "001 Входящие"])

    assert result.exit_code == 0
    assert "скудным" in result.output or "Firecrawl" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_cli.py -k ingest_url -v`
Expected: FAIL — `Error: No such command 'ingest-url'`.

- [ ] **Step 3: Implement**

В `knowledge-factory/kf/cli.py` добавить в блок импортов:

```python
from kf.web_extract import derive_filename, extract_from_url
```

Добавить новую команду (после команды `ingest`, перед `search`):

```python
@cli.command(name="ingest-url")
@click.argument("url")
@click.option("--dest", required=True, help="Раздел vault (относительно raw-data-repository), куда сохранить.")
def ingest_url(url: str, dest: str):
    """Скачать статью или видео по ссылке, сохранить в vault и сразу проиндексировать."""
    settings = load_settings()
    text, title, is_low_quality = extract_from_url(url, settings)

    dest_dir = DEFAULT_SOURCE / dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = derive_filename(title) + ".md"
    file_path = dest_dir / filename
    file_path.write_text(text, encoding="utf-8")

    click.echo(f"Сохранено: {dest}/{filename}")
    if is_low_quality:
        click.echo(
            "⚠ Извлечённый текст выглядит скудным — возможно, сайту нужен инструмент "
            "посильнее (Firecrawl, вручную)."
        )

    deps = _build_ingest_deps(settings)
    stats = ingest_directory(dest_dir, deps, detect_deletions=False)
    click.echo(f"Готово. проиндексировано: {stats.files_ingested}, чанков записано: {stats.chunks_written}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_cli.py -v`
Expected: PASS — все тесты файла, включая два новых.

- [ ] **Step 5: Commit**

```bash
git add "knowledge-factory/kf/cli.py" "knowledge-factory/tests/test_cli.py"
git commit -m "Digital brain | Команда kf.py ingest-url | V 1.4.52"
```

---

### Task 3: Аудио отдельным файлом

**Files:**
- Modify: `knowledge-factory/kf/extract.py`
- Modify: `knowledge-factory/kf/scope.py`
- Test: `knowledge-factory/tests/test_extract.py`
- Test: `knowledge-factory/tests/test_scope.py`

**Interfaces:**
- Consumes: `kf.transcribe.transcribe_audio` (существующая функция, без изменений).
- Produces: `kf.extract.extract_text()` теперь поддерживает `.mp3`, `.wav`, `.ogg`, `.m4a`;
  `kf.scope.should_index()` теперь пропускает эти расширения при сканировании.

- [ ] **Step 1: Write the failing tests**

Добавить в `knowledge-factory/tests/test_extract.py`:

```python
def test_dispatches_audio_file_to_transcription(tmp_path, monkeypatch):
    f = tmp_path / "voice.mp3"
    f.write_bytes(b"fakeaudio")
    monkeypatch.setattr(
        "kf.extract.transcribe_audio", lambda path, model_size, cache_dir: "Голосовое сообщение про отпуск"
    )

    text = extract_text(f, _dummy_settings())

    assert text == "Голосовое сообщение про отпуск"


def test_audio_extraction_supports_all_expected_extensions(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "kf.extract.transcribe_audio", lambda path, model_size, cache_dir: "текст"
    )

    for ext in (".mp3", ".wav", ".ogg", ".m4a"):
        f = tmp_path / f"file{ext}"
        f.write_bytes(b"fakeaudio")

        assert extract_text(f, _dummy_settings()) == "текст"
```

Добавить в `knowledge-factory/tests/test_scope.py` (файл уже существует — дописать в конец):

```python
def test_includes_audio_extensions(tmp_path):
    for ext in (".mp3", ".wav", ".ogg", ".m4a"):
        f = tmp_path / f"голосовое{ext}"
        f.write_bytes(b"x")

        assert should_index(f) is True
```

(Если в `test_scope.py` ещё нет импорта `should_index` из `kf.scope` — использовать тот же
импорт, что уже есть в начале файла для существующих тестов; ничего в существующих тестах не
менять.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_extract.py tests/test_scope.py -k audio -v`
Expected: FAIL — `ValueError: Unsupported file type: .mp3` (extract) и/или `assert False`
(scope, т.к. `.mp3` ещё не в `INCLUDED_EXTENSIONS`).

- [ ] **Step 3: Implement — `kf/extract.py`**

В `knowledge-factory/kf/extract.py` найти:

```python
from kf.ocr import extract_text_from_image
from kf.transcribe import transcribe_audio
from kf.video import extract_audio, sample_frames
from kf.vision_caption import caption_image

PLAIN_TEXT_EXTENSIONS = {".md", ".txt", ".csv", ".html"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
```

Заменить на:

```python
from kf.ocr import extract_text_from_image
from kf.transcribe import transcribe_audio
from kf.video import extract_audio, sample_frames
from kf.vision_caption import caption_image

PLAIN_TEXT_EXTENSIONS = {".md", ".txt", ".csv", ".html"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".m4a"}
```

Найти:

```python
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

Заменить на:

```python
def _extract_audio_file(path: Path, settings: Settings) -> str:
    return transcribe_audio(path, settings.whisper_model_size, settings.model_cache_dir)


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
    if suffix in AUDIO_EXTENSIONS:
        return _extract_audio_file(path, settings)
    raise ValueError(f"Unsupported file type: {suffix}")
```

- [ ] **Step 4: Implement — `kf/scope.py`**

Найти:

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

Заменить на:

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
    ".mp3",
    ".wav",
    ".ogg",
    ".m4a",
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_extract.py tests/test_scope.py -v`
Expected: PASS — все тесты обоих файлов.

- [ ] **Step 6: Commit**

```bash
git add "knowledge-factory/kf/extract.py" "knowledge-factory/kf/scope.py" "knowledge-factory/tests/test_extract.py" "knowledge-factory/tests/test_scope.py"
git commit -m "Digital brain | Поддержка аудио отдельным файлом (.mp3/.wav/.ogg/.m4a) | V 1.4.53"
```

---

### Task 4: Excel (.xlsx)

**Files:**
- Modify: `knowledge-factory/kf/extract.py`
- Modify: `knowledge-factory/kf/scope.py`
- Modify: `knowledge-factory/pyproject.toml` (добавить `openpyxl`)
- Test: `knowledge-factory/tests/test_extract.py`
- Test: `knowledge-factory/tests/test_scope.py`

**Interfaces:**
- Consumes: ничего нового.
- Produces: `kf.extract.extract_text()` теперь поддерживает `.xlsx`;
  `kf.scope.should_index()` теперь пропускает `.xlsx`.

- [ ] **Step 1: Add dependency**

Run: `cd "knowledge-factory" && uv add openpyxl`

- [ ] **Step 2: Write the failing tests**

Добавить в `knowledge-factory/tests/test_extract.py`:

```python
def test_extracts_xlsx_cells(tmp_path):
    import openpyxl

    f = tmp_path / "таблица.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Бюджет"
    sheet.append(["Статья", "Сумма"])
    sheet.append(["Реклама", 5000])
    workbook.save(f)

    text = extract_text(f, _dummy_settings())

    assert "Бюджет" in text
    assert "Статья" in text
    assert "Реклама" in text
    assert "5000" in text
```

Добавить в `knowledge-factory/tests/test_scope.py`:

```python
def test_includes_xlsx_extension(tmp_path):
    f = tmp_path / "таблица.xlsx"
    f.write_bytes(b"x")

    assert should_index(f) is True
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_extract.py tests/test_scope.py -k xlsx -v`
Expected: FAIL — `ValueError: Unsupported file type: .xlsx` (extract) и `assert False` (scope).

- [ ] **Step 4: Implement — `kf/extract.py`**

В `knowledge-factory/kf/extract.py` добавить в начало файла импорт:

```python
import openpyxl
```

Найти функцию `_extract_docx` и добавить сразу после неё:

```python
def _extract_xlsx(path: Path) -> str:
    workbook = openpyxl.load_workbook(path, data_only=True)
    parts = []
    for sheet in workbook.worksheets:
        parts.append(f"[Лист: {sheet.title}]")
        for row in sheet.iter_rows(values_only=True):
            cells = [str(cell) for cell in row if cell is not None]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)
```

Найти в `extract_text()`:

```python
    if suffix in AUDIO_EXTENSIONS:
        return _extract_audio_file(path, settings)
    raise ValueError(f"Unsupported file type: {suffix}")
```

Заменить на:

```python
    if suffix in AUDIO_EXTENSIONS:
        return _extract_audio_file(path, settings)
    if suffix == ".xlsx":
        return _extract_xlsx(path)
    raise ValueError(f"Unsupported file type: {suffix}")
```

- [ ] **Step 5: Implement — `kf/scope.py`**

Найти `INCLUDED_EXTENSIONS` (уже содержит аудио-расширения из Task 3) и добавить `.xlsx` в
множество:

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
    ".mp3",
    ".wav",
    ".ogg",
    ".m4a",
    ".xlsx",
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd "knowledge-factory" && uv run python -m pytest tests/test_extract.py tests/test_scope.py -v`
Expected: PASS — все тесты обоих файлов.

Затем прогнать полный набор тестов:

Run: `cd "knowledge-factory" && uv run python -m pytest -q`
Expected: все тесты PASS.

- [ ] **Step 7: Commit**

```bash
git add "knowledge-factory/kf/extract.py" "knowledge-factory/kf/scope.py" "knowledge-factory/pyproject.toml" "knowledge-factory/uv.lock" "knowledge-factory/tests/test_extract.py" "knowledge-factory/tests/test_scope.py"
git commit -m "Digital brain | Поддержка Excel-таблиц (.xlsx) | V 1.4.54"
```

---

### Task 5: Документация

**Files:**
- Modify: `E:\Digital brain\карта_бз.md`
- Modify: `E:\Digital brain\CLAUDE.md`
- Modify: `knowledge-factory/README.md`

Задача без тестов (документация) — TDD не применяется, отдельный шаг с коммитом.

- [ ] **Step 1: Обновить `E:\Digital brain\карта_бз.md`**

Найти абзац:

```markdown
Формат файла (PDF/docx/excel/ссылка/видео/аудио/голосовое/изображение и т.д.) определяет
только то, **как** материал обрабатывается технически при индексации (`kf/extract.py`:
текст читается напрямую, видео — транскрипция + разбор кадров, изображение — OCR/vision-caption
и т.д.) — не то, в какой раздел он попадает.
```

Заменить на:

```markdown
Формат файла (PDF/docx/xlsx/видео/аудио/изображение и т.д.) определяет только то, **как**
материал обрабатывается технически при индексации (`kf/extract.py`: текст читается напрямую,
видео — транскрипция + разбор кадров, изображение — OCR/vision-caption, xlsx — построчное
чтение ячеек и т.д.) — не то, в какой раздел он попадает.

**Ссылки** (на статьи и на YouTube-видео) — не читаются автоматически при простом добавлении
файла в vault, для них есть отдельная команда: `kf.py ingest-url <ссылка> --dest <раздел>`
(сама скачивает, извлекает текст через `trafilatura`/субтитры/Whisper и индексирует). Если
извлечённый текст статьи получился подозрительно скудным — команда предупредит об этом в
выводе; в этом случае ассистент вручную достаёт текст через Firecrawl и заменяет заметку.
```

- [ ] **Step 2: Обновить `E:\Digital brain\CLAUDE.md`**

Найти абзац:

```markdown
Если формат непонятен агенту (например, чтение конкретных ссылок/аудио ещё не
настроено) — прямо сказать об этом, а не притворяться, что материал обработан.
```

Заменить на:

```markdown
Ссылки на статьи и YouTube-видео обрабатываются командой `kf.py ingest-url <ссылка> --dest
<раздел>` (агент сам определяет раздел по той же таблице маршрутизации, как и для любого
другого материала, и передаёт его через `--dest`). Если команда предупредила, что извлечённый
текст статьи выглядит скудным — агент вручную достаёт текст через Firecrawl и подменяет им
содержимое сохранённого файла перед повторным `kf.py ingest`.

Если формат всё же непонятен агенту (видеохостинг без субтитров и не YouTube, сайт, который
не читается ни `trafilatura`, ни Firecrawl, и т.п.) — прямо сказать об этом пользователю, а
не притворяться, что материал обработан.
```

- [ ] **Step 3: Обновить `knowledge-factory/README.md`**

Найти блок:

```markdown
## Команды kf.py

```
uv run python kf.py ingest [--source PATH]   # индексация (по умолчанию — ../raw-data-repository)
uv run python kf.py search "запрос"          # семантический поиск фрагментов
uv run python kf.py ask "вопрос"             # поиск + связный ответ через OpenRouter, со ссылками
uv run python kf.py stats                    # сколько документов/чанков в базе
uv run python kf.py embedding-model list     # профили моделей эмбеддинга и их покрытие
uv run python kf.py embedding-model use <n>  # переключить активную модель (без переиндексации)
uv run python kf.py embedding-model sync     # досчитать недостающие эмбеддинги для активной модели
```
```

Заменить на:

```markdown
## Команды kf.py

```
uv run python kf.py ingest [--source PATH]   # индексация (по умолчанию — ../raw-data-repository)
uv run python kf.py ingest-url <url> --dest <раздел>   # скачать статью/YouTube-видео по ссылке и проиндексировать
uv run python kf.py search "запрос"          # семантический поиск фрагментов
uv run python kf.py ask "вопрос"             # поиск + связный ответ через OpenRouter, со ссылками
uv run python kf.py stats                    # сколько документов/чанков в базе
uv run python kf.py embedding-model list     # профили моделей эмбеддинга и их покрытие
uv run python kf.py embedding-model use <n>  # переключить активную модель (без переиндексации)
uv run python kf.py embedding-model sync     # досчитать недостающие эмбеддинги для активной модели
```
```

Найти раздел про исключённые файлы (`Закладки браузера — структура.md`) и сразу после него
добавить:

```markdown
`kf.py ingest`/`extract_text` теперь также читает отдельные аудиофайлы и голосовые записи
(`.mp3`/`.wav`/`.ogg`/`.m4a`, через тот же Whisper, что и для видео) и Excel-таблицы (`.xlsx`,
построчное чтение ячеек всех листов).
```

- [ ] **Step 4: Commit**

```bash
git add "карта_бз.md" "CLAUDE.md" "knowledge-factory/README.md"
git commit -m "Digital brain | Документация: ссылки, аудио, Excel в регламенте | V 1.4.55"
```
