from pathlib import Path

from kf.embedding_models import DEFAULT_PROFILE_NAME

STATE_FILENAME = "active_embedding_model.txt"


def _state_file(data_root: str) -> Path:
    return Path(data_root) / STATE_FILENAME


def get_active_profile_name(data_root: str) -> str:
    path = _state_file(data_root)
    if not path.exists():
        return DEFAULT_PROFILE_NAME
    name = path.read_text(encoding="utf-8").strip()
    return name or DEFAULT_PROFILE_NAME


def set_active_profile_name(data_root: str, name: str) -> None:
    path = _state_file(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name, encoding="utf-8")
