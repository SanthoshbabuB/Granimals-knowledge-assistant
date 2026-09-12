from pathlib import Path

from app.ingestion.pipeline import IngestionPipeline


PDF_PATH = Path(
    "data/documents/AtlasCopca-annual-report-2024.pdf"
)

ASSET_ROOT = Path(
    "data/assets"
)


def main() -> None:

    pipeline = IngestionPipeline(
        asset_root=ASSET_ROOT
    )

    chunks = pipeline.process(
        PDF_PATH
    )

    print("\n========== SAMPLE CHUNKS ==========\n")

    for chunk in chunks[:5]:
        print(
            f"ID: {chunk.chunk_id}"
        )
        print(
            f"Type: {chunk.content_type}"
        )
        print(
            f"Page: {chunk.page_number}"
        )
        print(
            f"Text:\n{chunk.text[:500]}"
        )
        print(
            "\n-----------------------------------\n"
        )

    print("\n========== OCR CHUNKS ==========\n")

    ocr_chunks = [
        chunk
        for chunk in chunks
        if chunk.content_type == "ocr"
    ]

    for chunk in ocr_chunks[:3]:
        print(
            f"Page: {chunk.page_number}"
        )
        print(
            f"Text:\n{chunk.text[:500]}"
        )
        print(
            "\n-----------------------------------\n"
        )

    print("\n========== TABLE CHUNKS ==========\n")

    table_chunks = [
        chunk
        for chunk in chunks
        if chunk.content_type == "table"
    ]

    for chunk in table_chunks[:3]:
        print(
            f"Page: {chunk.page_number}"
        )
        print(
            f"Text:\n{chunk.text[:500]}"
        )
        print(
            "\n-----------------------------------\n"
        )


if __name__ == "__main__":
    main()