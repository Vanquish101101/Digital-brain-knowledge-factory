from unittest.mock import MagicMock, patch

import pytest

from kf.web_extract import derive_filename, is_youtube_url, unique_filename, extract_from_url


def test_is_youtube_url_detects_standard_domain():
    assert is_youtube_url("https://www.youtube.com/watch?v=abc123") is True


def test_is_youtube_url_detects_short_domain():
    assert is_youtube_url("https://youtu.be/abc123") is True


def test_is_youtube_url_rejects_other_domains():
    assert is_youtube_url("https://example.com/article") is False


def test_is_youtube_url_rejects_lookalike_domain():
    assert is_youtube_url("https://notyoutube.com/watch?v=abc") is False


def test_is_youtube_url_rejects_domain_with_youtube_as_suffix_trick():
    assert is_youtube_url("https://youtube.com.evil.ru/watch?v=abc") is False


def test_is_youtube_url_accepts_subdomain():
    assert is_youtube_url("https://m.youtube.com/watch?v=abc") is True


def test_derive_filename_slugifies_title():
    assert derive_filename("Как настроить Docker: полное руководство!") == "Как-настроить-Docker-полное-руководство"


def test_derive_filename_truncates_long_title():
    long_title = "Слово " * 30
    result = derive_filename(long_title)

    assert len(result) <= 80


def test_derive_filename_falls_back_to_timestamp_when_no_title():
    result = derive_filename(None)

    assert result.startswith("url-")


def test_derive_filename_falls_back_to_timestamp_when_title_has_no_word_chars():
    result = derive_filename("!!!???")

    assert result.startswith("url-")


def test_unique_filename_returns_original_when_no_collision(tmp_path):
    assert unique_filename(tmp_path, "article.md") == "article.md"


def test_unique_filename_appends_suffix_on_collision(tmp_path):
    (tmp_path / "article.md").write_text("x", encoding="utf-8")

    assert unique_filename(tmp_path, "article.md") == "article-2.md"


def test_unique_filename_increments_past_multiple_collisions(tmp_path):
    (tmp_path / "article.md").write_text("x", encoding="utf-8")
    (tmp_path / "article-2.md").write_text("x", encoding="utf-8")

    assert unique_filename(tmp_path, "article.md") == "article-3.md"


def test_extract_from_url_uses_real_youtube_title(monkeypatch):
    monkeypatch.setattr("kf.web_extract.is_youtube_url", lambda url: True)
    monkeypatch.setattr("kf.web_extract.extract_youtube_transcript", lambda url: "текст транскрипта")
    monkeypatch.setattr("kf.web_extract._youtube_title", lambda url: "Настоящее Название Видео")

    text, title, is_low_quality = extract_from_url("https://youtube.com/watch?v=abc", MagicMock())

    assert title == "Настоящее Название Видео"
    assert text == "текст транскрипта"


def test_youtube_title_extracts_from_yt_dlp():
    from kf.web_extract import _youtube_title

    fake_ydl = MagicMock()
    fake_ydl.__enter__.return_value.extract_info.return_value = {"title": "Заголовок"}
    with patch("kf.web_extract.yt_dlp.YoutubeDL", return_value=fake_ydl):
        assert _youtube_title("https://youtube.com/watch?v=abc") == "Заголовок"


def test_youtube_title_returns_none_on_extraction_failure():
    from kf.web_extract import _youtube_title

    with patch("kf.web_extract.yt_dlp.YoutubeDL", side_effect=RuntimeError("network")):
        assert _youtube_title("https://youtube.com/watch?v=abc") is None
