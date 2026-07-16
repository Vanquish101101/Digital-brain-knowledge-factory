from kf.synthesize import build_synthesis_messages


def test_build_synthesis_messages_includes_path_and_text():
    messages = build_synthesis_messages("некий текст файла", "путь/файл.md")

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "путь/файл.md" in messages[0]["content"]
    assert "некий текст файла" in messages[0]["content"]


def test_build_synthesis_messages_asks_for_three_part_structure():
    messages = build_synthesis_messages("текст", "файл.md")

    content = messages[0]["content"]
    assert "ключевые идеи" in content.lower()
    assert "пригодиться" in content.lower()
