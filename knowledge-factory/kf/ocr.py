from pathlib import Path

import pytesseract
from PIL import Image


def extract_text_from_image(path: Path, languages: str = "rus+eng") -> str:
    image = Image.open(path)
    return pytesseract.image_to_string(image, lang=languages).strip()
