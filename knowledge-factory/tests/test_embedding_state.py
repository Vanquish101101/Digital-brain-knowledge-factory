from kf.embedding_state import get_active_profile_name, set_active_profile_name


def test_returns_default_when_no_state_file(tmp_path):
    assert get_active_profile_name(str(tmp_path)) == "local"


def test_set_then_get_roundtrip(tmp_path):
    set_active_profile_name(str(tmp_path), "qwen3-8b")

    assert get_active_profile_name(str(tmp_path)) == "qwen3-8b"


def test_set_overwrites_previous_value(tmp_path):
    set_active_profile_name(str(tmp_path), "qwen3-8b")
    set_active_profile_name(str(tmp_path), "openai-large")

    assert get_active_profile_name(str(tmp_path)) == "openai-large"


def test_get_strips_whitespace_from_state_file(tmp_path):
    state_dir = tmp_path
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "active_embedding_model.txt").write_text("  openai-small  \n", encoding="utf-8")

    assert get_active_profile_name(str(tmp_path)) == "openai-small"


def test_set_creates_data_root_if_missing(tmp_path):
    nested = tmp_path / "nested" / "data"

    set_active_profile_name(str(nested), "local")

    assert get_active_profile_name(str(nested)) == "local"
