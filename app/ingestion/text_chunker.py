from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class TextChunk:
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    text: str
    content_type: str = "text"

    image_id: str | None = None
    image_path: str | None = None
    table_id: str | None = None
    page_image_path: str | None = None


class TextChunker:
    """Create semantic-ish overlapping chunks from page text."""

    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 250,
    ) -> None:

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ", ",
                " ",
                "",
            ],
            length_function=len,
        )

    def chunk(
        self,
        document_id: str,
        document_name: str,
        page_number: int,
        text: str,
        content_type: str = "text",
    ) -> list[TextChunk]:

        if not text.strip():
            return []

        pieces = self.splitter.split_text(text)

        return [
            TextChunk(
                chunk_id=(
                    f"{document_id}"
                    f"_page_{page_number}"
                    f"_{content_type}"
                    f"_chunk_{index + 1}"
                ),
                document_id=document_id,
                document_name=document_name,
                page_number=page_number,
                text=piece,
                content_type=content_type,
            )
            for index, piece in enumerate(pieces)
        ]