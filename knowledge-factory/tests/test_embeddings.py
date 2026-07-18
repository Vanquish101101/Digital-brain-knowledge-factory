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


from kf.embedding_models import EMBEDDING_PROFILES
from kf.embeddings import build_embedding_request, embed_for_profile, get_embedder_for_profile, parse_embedding_response


def test_build_embedding_request_includes_model_and_texts():
    profile = EMBEDDING_PROFILES["openai-small"]

    request = build_embedding_request(profile, ["привет", "мир"])

    assert request == {"model": "openai/text-embedding-3-small", "input": ["привет", "мир"]}


def test_parse_embedding_response_extracts_vectors_in_order():
    raw = {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]}

    vectors = parse_embedding_response(raw)

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_get_embedder_for_profile_returns_none_for_openrouter_provider():
    settings = load_settings()
    profile = EMBEDDING_PROFILES["openai-small"]

    embedder = get_embedder_for_profile(settings, profile)

    assert embedder is None


def test_get_embedder_for_profile_returns_local_embedder_for_local_provider():
    settings = load_settings()
    profile = EMBEDDING_PROFILES["local"]

    embedder = get_embedder_for_profile(settings, profile)
    vectors = embed_for_profile(settings, profile, embedder, ["текст"])

    assert len(vectors) == 1
    assert len(vectors[0]) == 384
