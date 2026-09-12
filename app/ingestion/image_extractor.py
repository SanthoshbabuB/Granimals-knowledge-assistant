from pathlib import Path

import pymupdf

from app.ingestion.models import ImageAsset


class ImageExtractor:
    """Extract embedded images from PDF pages."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def extract_page_images(
        self,
        pdf_path: Path,
        document_id: str,
        document_name: str,
        page_number: int,
    ) -> list[ImageAsset]:

        image_output = (
            self.output_root
            / document_id
            / "images"
        )

        image_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        assets: list[ImageAsset] = []

        with pymupdf.open(pdf_path) as document:
            page = document[page_number - 1]

            images = page.get_images(
                full=True
            )

            for image_index, image_info in enumerate(images):
                xref = image_info[0]

                extracted = document.extract_image(
                    xref
                )

                extension = extracted["ext"]

                image_id = (
                    f"{document_id}"
                    f"_page_{page_number}"
                    f"_image_{image_index + 1}"
                )

                image_path = (
                    image_output
                    / f"{image_id}.{extension}"
                )

                image_path.write_bytes(
                    extracted["image"]
                )

                assets.append(
                    ImageAsset(
                        image_id=image_id,
                        document_id=document_id,
                        document_name=document_name,
                        page_number=page_number,
                        image_path=str(image_path),
                        image_type=extension,
                        width=extracted["width"],
                        height=extracted["height"],
                    )
                )

        return assets