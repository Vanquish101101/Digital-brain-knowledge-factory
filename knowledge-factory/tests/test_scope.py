from pathlib import Path

from kf.scope import should_index


def test_includes_markdown_note():
    assert should_index(Path("notes/idea.md")) is True


def test_includes_pdf():
    assert should_index(Path("docs/report.pdf")) is True


def test_includes_docx():
    assert should_index(Path("docs/report.docx")) is True


def test_includes_image():
    assert should_index(Path("photos/shot.png")) is True


def test_includes_video():
    assert should_index(Path("clips/intro.mp4")) is True


def test_excludes_archive():
    assert should_index(Path("backup/data.zip")) is False


def test_excludes_node_modules_dir():
    assert should_index(Path("app/node_modules/foo/bar.js")) is False


def test_excludes_git_dir():
    assert should_index(Path(".git/config")) is False


def test_excludes_venv_dir():
    assert should_index(Path("venv/lib/site-packages/x.py")) is False


def test_excludes_pycache_dir():
    assert should_index(Path("__pycache__/module.cpython-312.pyc")) is False


def test_excludes_bookmarks_dump_by_filename():
    path = Path("001 Входящие (Сырые данные)/Закладки браузера — структура.md")
    assert should_index(path) is False


def test_includes_audio_extensions(tmp_path):
    for ext in (".mp3", ".wav", ".ogg", ".m4a"):
        f = tmp_path / f"голосовое{ext}"
        f.write_bytes(b"x")

        assert should_index(f) is True
