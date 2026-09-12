from pathlib import Path

from PIL import Image, ImageDraw

from app.ingestion.ocr import OCRExtractor


def test_ocr_extractor() -> None:
    image_path = Path(
        "tests/test_ocr_image.png"
    )

    # Create a simple test image
    image = Image.new(
        "RGB",
        (800, 200),
        "white",
    )

    draw = ImageDraw.Draw(image)

    draw.text(
        (50, 70),
        "Atlas Copco Revenue 2024",
        fill="black",
    )

    image.save(image_path)

    # Create OCR extractor
    extractor = OCRExtractor(
        tesseract_path=(
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )
    )

    # Extract text
    extracted_text = extractor.extract(
        image_path
    )

    # Verify OCR worked
    assert isinstance(
        extracted_text,
        str,
    )

    assert "Atlas" in extracted_text
    assert "Copco" in extracted_text

    # Clean up test image
    image_path.unlink()