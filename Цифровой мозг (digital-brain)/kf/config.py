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
