import pytest

from kf.web_extract import derive_filename, is_youtube_url


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
