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
