from kf.llm import build_prompt


def test_includes_question_and_context_texts():
    contexts = [
        {"path": "notes/borscht.md", "text": "Борщ варится с говядиной и свёклой."},
        {"path": "notes/soup.md", "text": "Суп — это блюдо на бульоне."},
    ]

    messages = build_prompt("Как приготовить борщ?", contexts)

    assert messages[-1]["role"] == "user"
    assert "Как приготовить борщ?" in messages[-1]["content"]
    assert "Борщ варится с говядиной и свёклой." in messages[-1]["content"]
    assert "notes/borscht.md" in messages[-1]["content"]


def test_no_context_still_produces_valid_prompt():
    messages = build_prompt("Вопрос без контекста", [])

    assert any("Вопрос без контекста" in m["content"] for m in messages)
