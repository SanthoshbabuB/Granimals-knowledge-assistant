from pathlib import Path

import pymupdf

from app.ingestion.models import TableData


class TableExtractor:
    """Extract tables when supported by the PDF."""

    def extract_page_tables(
        self,
        pdf_path: Path,
        page_number: int,
    ) -> list[TableData]:

        tables: list[TableData] = []

        with pymupdf.open(pdf_path) as document:
            page = document[page_number - 1]

            if not hasattr(page, "find_tables"):
                return tables

            try:
                table_finder = page.find_tables()
            except Exception:
                return tables

            for table_index, table in enumerate(
                table_finder.tables
            ):
                rows = table.extract()

                normalized_rows = [
                    [
                        "" if cell is None else str(cell)
                        for cell in row
                    ]
                    for row in rows
                ]

                table_text = self._rows_to_text(
                    normalized_rows
                )

                tables.append(
                    TableData(
                        table_id=(
                            f"page_{page_number}"
                            f"_table_{table_index + 1}"
                        ),
                        page_number=page_number,
                        rows=normalized_rows,
                        text=table_text,
                    )
                )

        return tables

    @staticmethod
    def _rows_to_text(
        rows: list[list[str]],
    ) -> str:

        return "\n".join(
            " | ".join(row)
            for row in rows
        )