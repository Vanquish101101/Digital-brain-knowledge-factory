from kf.config import load_settings
from kf.embeddings import embed, get_embedder


def test_embed_returns_one_vector_per_text():
    settings = load_settings()
    embedder = get_embedder(settings)

    vectors = embed(embedder, ["привет мир", "hello world"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384


def test_different_texts_give_different_vectors():
    settings = load_settings()
    embedder = get_embedder(settings)

    vectors = embed(embedder, ["кошка", "космический корабль"])

    assert vectors[0] != vectors[1]
