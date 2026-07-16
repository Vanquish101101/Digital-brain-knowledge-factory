import httpx

from kf.config import Settings

SYNTHESIS_PROMPT_TEMPLATE = (
    "Ты помогаешь строить личную базу знаний пользователя. Ниже — текст, "
    "извлечённый из файла «{path}». Напиши структурированную заметку на "
    "русском языке из трёх частей:\n"
    "1. О чём этот материал (кратко, 1-2 предложения).\n"
    "2. Ключевые идеи и факты.\n"
    "3. Как это может пригодиться в будущих задачах и проектах.\n\n"
    "Текст:\n{text}"
)


def build_synthesis_messages(text: str, source_path: str) -> list[dict]:
    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(path=source_path, text=text)
    return [{"role": "user", "content": prompt}]


def synthesize_note(settings: Settings, text: str, source_path: str) -> str:
    messages = build_synthesis_messages(text, source_path)
    response = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
        json={"model": settings.llm_model, "messages": messages},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
