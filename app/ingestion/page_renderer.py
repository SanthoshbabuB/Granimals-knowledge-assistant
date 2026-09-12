from pathlib import Path

import pymupdf


class PageRenderer:
    """Render PDF pages to PNG images."""

    def __init__(
        self,
        output_root: Path,
        dpi: int = 150,
    ) -> None:
        self.output_root = output_root
        self.dpi = dpi

    def render_page(
        self,
        pdf_path: Path,
        document_id: str,
        page_number: int,
    ) -> Path:
        document_output = (
            self.output_root / document_id / "pages"
        )

        document_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            document_output
            / f"page_{page_number:03d}.png"
        )

        with pymupdf.open(pdf_path) as document:
            page = document[page_number - 1]

            zoom = self.dpi / 72

            matrix = pymupdf.Matrix(
                zoom,
                zoom,
            )

            pixmap = page.get_pixmap(
                matrix=matrix,
                alpha=False,
            )

            pixmap.save(output_path)

        return output_path