from fastembed import TextEmbedding
import httpx

from kf.config import Settings
from kf.embedding_models import EmbeddingProfile


def get_embedder(settings: Settings) -> TextEmbedding:
    return TextEmbedding(
        model_name=settings.embedding_model,
        cache_dir=settings.model_cache_dir,
    )


def embed(embedder: TextEmbedding, texts: list[str]) -> list[list[float]]:
    return [vec.tolist() for vec in embedder.embed(texts)]


def build_embedding_request(profile: EmbeddingProfile, texts: list[str]) -> dict:
    return {"model": profile.model_id, "input": texts}


def parse_embedding_response(raw_json: dict) -> list[list[float]]:
    return [item["embedding"] for item in raw_json["data"]]


def embed_via_openrouter(settings: Settings, profile: EmbeddingProfile, texts: list[str]) -> list[list[float]]:
    response = httpx.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
        json=build_embedding_request(profile, texts),
        timeout=60,
    )
    response.raise_for_status()
    return parse_embedding_response(response.json())


def get_embedder_for_profile(settings: Settings, profile: EmbeddingProfile):
    if profile.provider != "local":
        return None
    return TextEmbedding(model_name=profile.model_id, cache_dir=settings.model_cache_dir)


def embed_for_profile(
    settings: Settings, profile: EmbeddingProfile, embedder, texts: list[str]
) -> list[list[float]]:
    if profile.provider == "local":
        return embed(embedder, texts)
    return embed_via_openrouter(settings, profile, texts)
