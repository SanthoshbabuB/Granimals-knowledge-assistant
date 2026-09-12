import json
from pathlib import Path

from app.ingestion.text_chunker import TextChunk


class ChunkStorage:
    """Save and load processed chunks as JSONL."""

    def save(
        self,
        chunks: list[TextChunk],
        output_path: Path,
    ) -> None:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            for chunk in chunks:

                record = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                    "text": chunk.text,
                    "content_type": chunk.content_type,
                    "image_id": chunk.image_id,
                    "image_path": chunk.image_path,
                    "table_id": chunk.table_id,
                    "page_image_path": chunk.page_image_path,
                }

                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def load(
        self,
        input_path: Path,
    ) -> list[TextChunk]:

        if not input_path.exists():
            raise FileNotFoundError(
                f"Chunk file does not exist: {input_path}"
            )

        chunks: list[TextChunk] = []

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                record = json.loads(line)

                chunks.append(
                    TextChunk(
                        chunk_id=record["chunk_id"],
                        document_id=record["document_id"],
                        document_name=record["document_name"],
                        page_number=record["page_number"],
                        text=record["text"],
                        content_type=record["content_type"],
                        image_id=record.get("image_id"),
                        image_path=record.get("image_path"),
                        table_id=record.get("table_id"),
                        page_image_path=record.get(
                            "page_image_path"
                        ),
                    )
                )

        return chunks