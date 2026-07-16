from kf.journal import append_entries, detect_deleted, extract_description, format_entry


def test_extract_description_strips_leading_numbering():
    note = "1. Материал про Docker и контейнеризацию.\n2. Ключевые идеи..."

    assert extract_description(note) == "Материал про Docker и контейнеризацию."


def test_extract_description_strips_leading_dash():
    note = "- Курс по 3D-моделированию в Blender.\nПодробности дальше."

    assert extract_description(note) == "Курс по 3D-моделированию в Blender."


def test_extract_description_preserves_leading_digit_that_is_not_numbering():
    note = "3D-моделирование в Blender.\nПодробности дальше."

    assert extract_description(note) == "3D-моделирование в Blender."


def test_extract_description_truncates_long_first_line():
    note = "x" * 200

    assert extract_description(note) == "x" * 150


def test_extract_description_empty_for_blank_note():
    assert extract_description("") == ""
    assert extract_description("   \n  ") == ""


def test_format_entry_with_description():
    line = format_entry(
        "добавлено", "005 Ресурсы/курс.pdf", "005 Ресурсы", "О курсе по 3D", "2026-07-16"
    )

    assert line == "- 2026-07-16 | добавлено | 005 Ресурсы/курс.pdf | 005 Ресурсы | О курсе по 3D"


def test_format_entry_without_description():
    line = format_entry("удалено", "003 Знания/старое.md", "003 Знания", "", "2026-07-16")

    assert line == "- 2026-07-16 | удалено | 003 Знания/старое.md | 003 Знания | (без описания)"


def test_detect_deleted_finds_paths_missing_from_seen():
    known = {"a.md", "b.md", "c.md"}
    seen = {"a.md", "c.md"}

    assert detect_deleted(known, seen) == {"b.md"}


def test_detect_deleted_empty_when_nothing_missing():
    known = {"a.md"}
    seen = {"a.md"}

    assert detect_deleted(known, seen) == set()


def test_append_entries_creates_file_with_header(tmp_path):
    journal = tmp_path / "Журнал знаний.md"

    append_entries(["- entry one"], journal)

    content = journal.read_text(encoding="utf-8")
    assert "# Журнал знаний" in content
    assert "- entry one" in content


def test_append_entries_appends_to_existing_file(tmp_path):
    journal = tmp_path / "Журнал знаний.md"
    append_entries(["- entry one"], journal)

    append_entries(["- entry two"], journal)

    content = journal.read_text(encoding="utf-8")
    assert "- entry one" in content
    assert "- entry two" in content


def test_append_entries_noop_for_empty_list(tmp_path):
    journal = tmp_path / "Журнал знаний.md"

    append_entries([], journal)

    assert not journal.exists()
