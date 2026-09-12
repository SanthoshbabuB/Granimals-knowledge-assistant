from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

import pymupdf

from app.ingestion.models import DocumentData, PageData


class PDFLoader:
    """Load page-level text and basic PDF structure."""

    def load(self, file_path: Path) -> DocumentData:
        if not file_path.exists():
            raise FileNotFoundError(
                f"PDF file does not exist: {file_path}"
            )

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Unsupported file type: {file_path.suffix}"
            )

        document_id = str(
            uuid5(
                NAMESPACE_URL,
                str(file_path.resolve()),
            )
        )

        pages: list[PageData] = []

        with pymupdf.open(file_path) as document:
            for page_index, page in enumerate(document):
                page_number = page_index + 1

                text = page.get_text("text")

                pages.append(
                    PageData(
                        document_id=document_id,
                        document_name=file_path.name,
                        page_number=page_number,
                        text=text,
                    )
                )

        return DocumentData(
            document_id=document_id,
            document_name=file_path.name,
            source_path=file_path,
            pages=pages,
        )