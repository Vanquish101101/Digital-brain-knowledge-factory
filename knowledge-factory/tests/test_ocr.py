from PIL import Image, ImageDraw, ImageFont

from kf.ocr import extract_text_from_image


def test_recognizes_text_in_generated_image(tmp_path):
    img = Image.new("RGB", (400, 120), color="white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 40)
    draw.text((10, 30), "HELLO WORLD", fill="black", font=font)
    f = tmp_path / "screenshot.png"
    img.save(f)

    text = extract_text_from_image(f, languages="eng")

    assert "HELLO" in text.upper()
