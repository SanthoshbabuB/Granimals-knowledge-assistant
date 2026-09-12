from pathlib import Path

from app.core.config import get_settings
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.storage import ChunkStorage


DOCUMENTS_DIR = Path("data/documents")
ASSET_ROOT = Path("data/assets")
PROCESSED_ROOT = Path("data/processed")


def main() -> None:
    """
    Ingest every PDF in data/documents/.

    No document names are hardcoded.
    """

    settings = get_settings()

    # ---------------------------------------------------------
    # Validate directories
    # ---------------------------------------------------------

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {DOCUMENTS_DIR}"
        )

    ASSET_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Discover PDFs
    # ---------------------------------------------------------

    pdf_files = sorted(
        DOCUMENTS_DIR.glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DOCUMENTS_DIR}"
        )

    print("\n" + "=" * 80)
    print("MULTI-DOCUMENT INGESTION")
    print("=" * 80)

    print(
        f"Found {len(pdf_files)} PDF(s)"
    )

    for pdf_file in pdf_files:
        print(
            f"  - {pdf_file.name}"
        )

    print("=" * 80)

    # ---------------------------------------------------------
    # Create ingestion pipeline
    # ---------------------------------------------------------

    pipeline = IngestionPipeline(
        asset_root=ASSET_ROOT,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        enable_ocr=settings.enable_ocr,
        enable_vision=settings.enable_vision,
    )

    storage = ChunkStorage()

    # ---------------------------------------------------------
    # Process every PDF
    # ---------------------------------------------------------

    total_chunks = 0

    for index, pdf_file in enumerate(
        pdf_files,
        start=1,
    ):

        print("\n" + "=" * 80)
        print(
            f"DOCUMENT {index}/{len(pdf_files)}"
        )
        print("=" * 80)

        print(
            f"File: {pdf_file.name}"
        )

        # -----------------------------------------------------
        # Ingest document
        # -----------------------------------------------------

        document_data, chunks = pipeline.process(
            pdf_file
        )

        # -----------------------------------------------------
        # Create output filename
        # -----------------------------------------------------

        output_name = (
            f"{pdf_file.stem}_chunks.jsonl"
        )

        output_file = (
            PROCESSED_ROOT / output_name
        )

        # -----------------------------------------------------
        # Save chunks
        # -----------------------------------------------------

        storage.save(
            chunks,
            output_file,
        )

        print(
            f"Saved {len(chunks)} chunks to:"
        )

        print(
            f"  {output_file}"
        )

        total_chunks += len(chunks)

        # -----------------------------------------------------
        # Document summary
        # -----------------------------------------------------

        print("\nDocument summary:")

        print(
            f"  Document ID: "
            f"{document_data.document_id}"
        )

        print(
            f"  Pages: "
            f"{len(document_data.pages)}"
        )

        print(
            f"  Chunks: "
            f"{len(chunks)}"
        )

        image_count = sum(
            len(page.images)
            for page in document_data.pages
        )

        table_count = sum(
            len(page.tables)
            for page in document_data.pages
        )

        print(
            f"  Images: "
            f"{image_count}"
        )

        print(
            f"  Tables: "
            f"{table_count}"
        )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("INGESTION COMPLETE")
    print("=" * 80)

    print(
        f"Documents processed: "
        f"{len(pdf_files)}"
    )

    print(
        f"Total chunks created: "
        f"{total_chunks}"
    )

    print(
        f"Processed output: "
        f"{PROCESSED_ROOT}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()