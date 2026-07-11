from pathlib import Path

from docx import Document
from pypdf import PdfReader

PLAIN_TEXT_EXTENSIONS = {".md", ".txt", ".csv", ".html"}


def _extract_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text)


def _extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in PLAIN_TEXT_EXTENSIONS:
        return path.read_text(encoding="utf-8")
    if suffix == ".docx":
        return _extract_docx(path)
    if suffix == ".pdf":
        return _extract_pdf(path)
    raise ValueError(f"Unsupported file type: {suffix}")
