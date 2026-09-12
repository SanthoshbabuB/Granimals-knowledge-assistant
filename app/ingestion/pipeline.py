from pathlib import Path

from app.ingestion.cleaner import TextCleaner
from app.ingestion.image_extractor import ImageExtractor
from app.ingestion.loader import PDFLoader
from app.ingestion.models import DocumentData
from app.ingestion.ocr import OCRExtractor
from app.ingestion.page_renderer import PageRenderer
from app.ingestion.table_extractor import TableExtractor
from app.ingestion.text_chunker import TextChunker


class IngestionPipeline:
    """
    Document ingestion pipeline.

    Supports multiple independent PDF documents.

    Each PDF is processed independently and produces:
    - page-level text
    - tables
    - embedded images
    - rendered pages
    - OCR content
    - semantic text chunks
    """

    def __init__(
        self,
        asset_root: Path,
        chunk_size: int = 1500,
        chunk_overlap: int = 250,
        enable_ocr: bool = True,
        enable_vision: bool = True,
    ) -> None:

        self.asset_root = asset_root
        self.enable_ocr = enable_ocr
        self.enable_vision = enable_vision

        self.pdf_loader = PDFLoader()

        # PageRenderer expects output_root, not asset_root
        self.page_renderer = PageRenderer(
            output_root=asset_root
        )

        self.image_extractor = ImageExtractor(
            output_root=asset_root
        )

        self.table_extractor = TableExtractor()

        self.text_cleaner = TextCleaner()

        self.text_chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.ocr_extractor = OCRExtractor()

    def process(self, pdf_path: Path):
        """
        Process one PDF document.

        Returns:
            document: DocumentData
            chunks: list[TextChunk]
        """

        print(f"\nProcessing document: {pdf_path.name}")

        # ---------------------------------------------------------
        # 1. Load PDF
        # ---------------------------------------------------------
        document = self.pdf_loader.load(pdf_path)

        print(f"Loaded {len(document.pages)} pages")

        # ---------------------------------------------------------
        # 2. Process each page
        # ---------------------------------------------------------
        for page in document.pages:

            print(f"\nProcessing page {page.page_number}")

            # -----------------------------------------------------
            # 2.1 Render complete page as PNG
            # -----------------------------------------------------
            rendered_page = self.page_renderer.render_page(
                pdf_path=pdf_path,
                page_number=page.page_number,
                document_id=document.document_id,
            )

            page.rendered_page_path = str(rendered_page)

            # -----------------------------------------------------
            # 2.2 Extract embedded images
            # -----------------------------------------------------
            images = self.image_extractor.extract_page_images(
                pdf_path=pdf_path,
                page_number=page.page_number,
                document_id=document.document_id,
                document_name=document.document_name,
            )

            page.images.extend(images)

            # -----------------------------------------------------
            # 2.3 Extract tables
            # -----------------------------------------------------
            tables = self.table_extractor.extract_page_tables(
                pdf_path=pdf_path,
                page_number=page.page_number,
            )

            page.tables.extend(tables)

            # -----------------------------------------------------
            # 2.4 Clean page text
            # -----------------------------------------------------
            page.text = self.text_cleaner.clean(
                page.text
            )

            # -----------------------------------------------------
            # 2.5 OCR extracted images
            # -----------------------------------------------------
            if self.enable_ocr:
                self._process_ocr(page)

        # ---------------------------------------------------------
        # 3. Create chunks
        # ---------------------------------------------------------
        chunks = []

        for page in document.pages:

            page_chunks = self.text_chunker.chunk(
                document_id=document.document_id,
                document_name=document.document_name,
                page_number=page.page_number,
                text=page.text,
                content_type="text",
            )

            # Add page image information to chunks
            for chunk in page_chunks:
                chunk.page_image_path = page.rendered_page_path

            chunks.extend(page_chunks)

            # -----------------------------------------------------
            # Add table chunks
            # -----------------------------------------------------
            for table in page.tables:

                table_chunks = self.text_chunker.chunk(
                    document_id=document.document_id,
                    document_name=document.document_name,
                    page_number=page.page_number,
                    text=table.text,
                    content_type="table",
                )

                for chunk in table_chunks:
                    chunk.table_id = table.table_id
                    chunk.page_image_path = page.rendered_page_path

                chunks.extend(table_chunks)

            # -----------------------------------------------------
            # Add OCR chunks
            # -----------------------------------------------------
            for image in page.images:

                if not image.ocr_text:
                    continue

                ocr_chunks = self.text_chunker.chunk(
                    document_id=document.document_id,
                    document_name=document.document_name,
                    page_number=page.page_number,
                    text=image.ocr_text,
                    content_type="ocr",
                )

                for chunk in ocr_chunks:
                    chunk.image_id = image.image_id
                    chunk.image_path = image.image_path
                    chunk.page_image_path = page.rendered_page_path

                chunks.extend(ocr_chunks)

        print(f"\nCreated {len(chunks)} chunks")

        return document, chunks

    def _process_ocr(self, page) -> None:
        """
        Extract OCR text from embedded images on a page.
        """

        if not page.images:
            return

        for image in page.images:

            try:
                ocr_text = self.ocr_extractor.extract(
                    image.image_path
                )

                if ocr_text:
                    image.ocr_text = ocr_text.strip()

            except Exception as exc:
                print(
                    f"OCR failed for "
                    f"{image.image_path}: {exc}"
                )