from pathlib import Path

EXCLUDED_DIR_NAMES = {
    "node_modules",
    ".git",
    "venv",
    ".venv",
    "dist",
    "build",
    "__pycache__",
}

INCLUDED_EXTENSIONS = {
    ".md",
    ".txt",
    ".pdf",
    ".docx",
    ".csv",
    ".html",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".mp3",
    ".wav",
    ".ogg",
    ".m4a",
}

# Отдельные файлы, вручную исключённые из индексации (низкая ценность для поиска,
# несоразмерно большой объём). См. project-config/РЕШЕНИЯ-И-СТАТУС.md
EXCLUDED_FILENAMES = {
    "Закладки браузера — структура.md",
}


def should_index(path: Path) -> bool:
    if EXCLUDED_DIR_NAMES & set(path.parts):
        return False
    if path.name in EXCLUDED_FILENAMES:
        return False
    return path.suffix.lower() in INCLUDED_EXTENSIONS
