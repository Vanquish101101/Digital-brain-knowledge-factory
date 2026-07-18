import pytest

from kf.embedding_models import DEFAULT_PROFILE_NAME, EMBEDDING_PROFILES, get_profile


def test_default_profile_is_local_and_matches_existing_collection():
    assert DEFAULT_PROFILE_NAME == "local"
    local = EMBEDDING_PROFILES["local"]
    assert local.provider == "local"
    assert local.collection == "knowledge"
    assert local.dimension == 384


def test_all_expected_profiles_are_registered():
    assert set(EMBEDDING_PROFILES.keys()) == {"local", "qwen3-8b", "openai-small", "openai-large"}


def test_openrouter_profiles_have_openrouter_provider():
    for name in ("qwen3-8b", "openai-small", "openai-large"):
        assert EMBEDDING_PROFILES[name].provider == "openrouter"


def test_collection_names_are_unique():
    collections = [p.collection for p in EMBEDDING_PROFILES.values()]
    assert len(collections) == len(set(collections))


def test_get_profile_returns_matching_profile():
    profile = get_profile("openai-small")
    assert profile.model_id == "openai/text-embedding-3-small"
    assert profile.dimension == 1536


def test_get_profile_raises_on_unknown_name():
    with pytest.raises(ValueError):
        get_profile("не-существует")
