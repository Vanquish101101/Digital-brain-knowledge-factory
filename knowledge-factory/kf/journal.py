import re
from pathlib import Path

JOURNAL_HEADER = (
    "# Журнал знаний\n\n"
    "Автоматический лог добавлений, изменений и удалений в raw-data-repository/, "
    "пополняется при каждом запуске kf.py ingest.\n\n"
)

_NUMBERING_PATTERN = re.compile(r"^\s*(?:\d+[.)]|[-*#])\s*")


def extract_description(note_text: str) -> str:
    stripped = note_text.strip()
    if not stripped:
        return ""
    first_line = stripped.splitlines()[0]
    cleaned = _NUMBERING_PATTERN.sub("", first_line, count=1)
    return cleaned[:150]


def format_entry(action: str, path: str, section: str, description: str, date: str) -> str:
    desc = description if description else "(без описания)"
    return f"- {date} | {action} | {path} | {section} | {desc}"


def detect_deleted(known_paths: set[str], seen_paths: set[str]) -> set[str]:
    return known_paths - seen_paths


def append_entries(entries: list[str], path: Path) -> None:
    if not entries:
        return
    if not path.exists():
        path.write_text(JOURNAL_HEADER, encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry + "\n")
