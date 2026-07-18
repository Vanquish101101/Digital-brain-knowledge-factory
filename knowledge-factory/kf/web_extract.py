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
    return host == "youtube.com" or host.endswith(".youtube.com") or host == "youtu.be"


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
