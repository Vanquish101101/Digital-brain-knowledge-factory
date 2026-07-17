import json

import pytest

from kf.graph import build_extraction_messages, parse_extraction_response


def test_build_extraction_messages_includes_path_and_text():
    messages = build_extraction_messages("текст файла про Blender", "007 Проекты/файл.md")

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "007 Проекты/файл.md" in messages[0]["content"]
    assert "текст файла про Blender" in messages[0]["content"]


def test_build_extraction_messages_lists_fixed_categories():
    messages = build_extraction_messages("текст", "файл.md")

    content = messages[0]["content"]
    for category in ["использует", "часть_проекта", "связано_с_темой", "автор_создатель", "другое"]:
        assert category in content


def test_parse_extraction_response_returns_entities_and_relationships():
    raw = json.dumps(
        {
            "entities": [{"name": "Blender", "type": "инструмент"}, {"name": "Проект X", "type": "проект"}],
            "relationships": [
                {"from": "Blender", "to": "Проект X", "category": "использует", "description": "рендеринг сцен"}
            ],
        }
    )

    entities, relationships = parse_extraction_response(raw)

    assert entities == [{"name": "Blender", "type": "инструмент"}, {"name": "Проект X", "type": "проект"}]
    assert relationships == [
        {"from": "Blender", "to": "Проект X", "category": "использует", "description": "рендеринг сцен"}
    ]


def test_parse_extraction_response_raises_on_invalid_json():
    with pytest.raises(ValueError):
        parse_extraction_response("это не json вообще")


def test_parse_extraction_response_raises_on_missing_top_level_keys():
    with pytest.raises(ValueError):
        parse_extraction_response(json.dumps({"entities": []}))


def test_parse_extraction_response_raises_on_entity_missing_fields():
    raw = json.dumps({"entities": [{"name": "Blender"}], "relationships": []})

    with pytest.raises(ValueError):
        parse_extraction_response(raw)


def test_parse_extraction_response_raises_on_relationship_missing_fields():
    raw = json.dumps(
        {"entities": [], "relationships": [{"from": "A", "to": "B", "category": "другое"}]}
    )

    with pytest.raises(ValueError):
        parse_extraction_response(raw)
