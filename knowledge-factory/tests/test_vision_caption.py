from kf.vision_caption import build_vision_messages


def test_builds_message_with_base64_image_and_png_mime(tmp_path):
    f = tmp_path / "photo.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\nfakepngbytes")

    messages = build_vision_messages(f)

    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_uses_jpeg_mime_for_jpg_extension(tmp_path):
    f = tmp_path / "photo.jpg"
    f.write_bytes(b"fakejpegbytes")

    messages = build_vision_messages(f)

    assert messages[0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
