import base64
from pathlib import Path

import httpx

from kf.config import Settings

_MIME_BY_SUFFIX = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}

_DEFAULT_PROMPT = "Опиши, что изображено на картинке, подробно и по-русски."


def build_vision_messages(image_path: Path, prompt: str = _DEFAULT_PROMPT) -> list[dict]:
    image_bytes = Path(image_path).read_bytes()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    suffix = Path(image_path).suffix.lower().lstrip(".")
    mime = _MIME_BY_SUFFIX.get(suffix, "image/png")
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }
    ]


def caption_image(settings: Settings, image_path: Path) -> str:
    messages = build_vision_messages(image_path)
    response = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
        json={"model": settings.vision_model, "messages": messages},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
