import hashlib

from kf.hashing import sha256_of_file


def test_returns_correct_sha256(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("привет мир", encoding="utf-8")

    expected = hashlib.sha256("привет мир".encode("utf-8")).hexdigest()

    assert sha256_of_file(f) == expected


def test_different_content_gives_different_hash(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("a", encoding="utf-8")
    f2.write_text("b", encoding="utf-8")

    assert sha256_of_file(f1) != sha256_of_file(f2)
