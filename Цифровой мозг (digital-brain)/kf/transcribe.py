from pathlib import Path

from faster_whisper import WhisperModel


def transcribe_audio(path: Path, model_size: str = "small", cache_dir: str = "./data/model-cache") -> str:
    model = WhisperModel(model_size, device="cpu", compute_type="int8", download_root=cache_dir)
    segments, _ = model.transcribe(str(path), language="en")
    return " ".join(segment.text.strip() for segment in segments).strip()
