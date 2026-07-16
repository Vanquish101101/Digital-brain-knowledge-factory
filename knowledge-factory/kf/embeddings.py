from fastembed import TextEmbedding

from kf.config import Settings


def get_embedder(settings: Settings) -> TextEmbedding:
    return TextEmbedding(
        model_name=settings.embedding_model,
        cache_dir=settings.model_cache_dir,
    )


def embed(embedder: TextEmbedding, texts: list[str]) -> list[list[float]]:
    return [vec.tolist() for vec in embedder.embed(texts)]
