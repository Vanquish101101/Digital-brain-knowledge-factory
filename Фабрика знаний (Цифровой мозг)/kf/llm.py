import httpx

from kf.config import Settings

SYSTEM_PROMPT = (
    "Ты — ассистент по личной базе знаний пользователя. "
    "Отвечай только на основе предоставленных фрагментов. "
    "Если ответа нет во фрагментах — прямо скажи, что не нашёл. "
    "В конце ответа перечисли пути файлов-источников."
)


def build_prompt(question: str, contexts: list[dict]) -> list[dict]:
    if contexts:
        context_block = "\n\n".join(
            f"[{c['path']}]\n{c['text']}" for c in contexts
        )
    else:
        context_block = "(ничего не найдено в базе знаний)"

    user_content = f"Вопрос: {question}\n\nНайденные фрагменты:\n{context_block}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def call_llm(settings: Settings, messages: list[dict]) -> str:
    response = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
        json={"model": settings.llm_model, "messages": messages},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
