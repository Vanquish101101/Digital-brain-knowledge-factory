import json

import httpx

from kf.config import Settings

ENTITY_CATEGORIES = ["использует", "часть_проекта", "связано_с_темой", "автор_создатель", "другое"]

EXTRACTION_PROMPT_TEMPLATE = (
    "Ты строишь граф знаний личной базы пользователя. Ниже — текст, извлечённый из файла "
    "«{path}». Извлеки из текста сущности (люди, инструменты, проекты, темы, концепты) и "
    "связи между ними.\n\n"
    "Верни СТРОГО валидный JSON без пояснений и без markdown-разметки, в формате:\n"
    '{{"entities": [{{"name": "...", "type": "..."}}], '
    '"relationships": [{{"from": "...", "to": "...", "category": "...", "description": "..."}}]}}\n\n'
    "Поле category у каждой связи должно быть ровно одним из значений: "
    "использует, часть_проекта, связано_с_темой, автор_создатель, другое.\n\n"
    "Текст:\n{text}"
)


def build_extraction_messages(text: str, source_path: str) -> list[dict]:
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(path=source_path, text=text)
    return [{"role": "user", "content": prompt}]


def parse_extraction_response(raw_response: str) -> tuple[list[dict], list[dict]]:
    try:
        data = json.loads(raw_response)
        entities = data["entities"]
        relationships = data["relationships"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"не удалось разобрать ответ извлечения сущностей: {exc}") from exc

    for entity in entities:
        if "name" not in entity or "type" not in entity:
            raise ValueError("сущность без обязательных полей name/type в ответе извлечения")
    for rel in relationships:
        if not all(key in rel for key in ("from", "to", "category", "description")):
            raise ValueError("связь без обязательных полей from/to/category/description в ответе извлечения")

    return entities, relationships


def extract_entities_and_relationships(
    settings: Settings, text: str, source_path: str
) -> tuple[list[dict], list[dict]]:
    messages = build_extraction_messages(text, source_path)
    response = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
        json={"model": settings.llm_model, "messages": messages},
        timeout=60,
    )
    response.raise_for_status()
    raw_content = response.json()["choices"][0]["message"]["content"]
    return parse_extraction_response(raw_content)
