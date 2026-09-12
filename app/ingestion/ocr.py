from pathlib import Path

import pytesseract
from PIL import Image


class OCRExtractor:
    """Extract text from images using Tesseract OCR."""

    def __init__(
        self,
        tesseract_path: str | None = None,
    ) -> None:

        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = (
                tesseract_path
            )

    def extract(
        self,
        image_path: Path,
    ) -> str:

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image does not exist: {image_path}"
            )

        with Image.open(image_path) as image:

            text = pytesseract.image_to_string(
                image
            )

        return text.strip()

    def should_process(
        self,
        image_path: Path,
    ) -> bool:
        """Decide whether an image is worth sending to OCR."""

        if not image_path.exists():
            return False

        with Image.open(image_path) as image:

            width, height = image.size

        # Ignore extremely small images/icons.
        if width < 100 or height < 100:
            return False

        return True