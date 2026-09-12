from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TableData:
    table_id: str
    page_number: int
    rows: list[list[str]]
    text: str


@dataclass
class ImageAsset:
    image_id: str
    document_id: str
    document_name: str
    page_number: int
    image_path: str
    image_type: str
    width: int
    height: int
    caption: str | None = None
    ocr_text: str | None = None
    bounding_box: tuple[float, float, float, float] | None = None


@dataclass
class PageData:
    document_id: str
    document_name: str
    page_number: int
    text: str
    tables: list[TableData] = field(default_factory=list)
    images: list[ImageAsset] = field(default_factory=list)
    rendered_page_path: str | None = None


@dataclass
class DocumentData:
    document_id: str
    document_name: str
    source_path: Path
    pages: list[PageData] = field(default_factory=list)