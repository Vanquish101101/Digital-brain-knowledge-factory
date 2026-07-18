from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingProfile:
    name: str
    provider: str
    model_id: str
    dimension: int
    collection: str


DEFAULT_PROFILE_NAME = "local"

EMBEDDING_PROFILES: dict[str, EmbeddingProfile] = {
    "local": EmbeddingProfile(
        name="local",
        provider="local",
        model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        dimension=384,
        collection="knowledge",
    ),
    "qwen3-8b": EmbeddingProfile(
        name="qwen3-8b",
        provider="openrouter",
        model_id="qwen/qwen3-embedding-8b",
        dimension=4096,
        collection="knowledge__qwen3_8b",
    ),
    "openai-small": EmbeddingProfile(
        name="openai-small",
        provider="openrouter",
        model_id="openai/text-embedding-3-small",
        dimension=1536,
        collection="knowledge__openai_small",
    ),
    "openai-large": EmbeddingProfile(
        name="openai-large",
        provider="openrouter",
        model_id="openai/text-embedding-3-large",
        dimension=3072,
        collection="knowledge__openai_large",
    ),
}


def get_profile(name: str) -> EmbeddingProfile:
    if name not in EMBEDDING_PROFILES:
        raise ValueError(f"неизвестный профиль эмбеддинга: {name}")
    return EMBEDDING_PROFILES[name]
