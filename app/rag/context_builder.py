from dataclasses import dataclass


@dataclass
class ContextItem:
    """A retrieved piece of evidence."""

    chunk_id: str
    document_name: str
    page_number: int
    content_type: str
    text: str
    image_path: str | None = None
    page_image_path: str | None = None


class ContextBuilder:
    """Convert reranked retrieval results into LLM context."""

    def build(self, reranked_results: list) -> list[ContextItem]:
        context_items: list[ContextItem] = []

        for result in reranked_results:
            original_result = result["result"]

            if isinstance(original_result, dict):
                chunk = original_result["chunk"]

                context_items.append(
                    ContextItem(
                        chunk_id=chunk.chunk_id,
                        document_name=chunk.document_name,
                        page_number=chunk.page_number,
                        content_type=chunk.content_type,
                        text=chunk.text,
                        image_path=chunk.image_path,
                        page_image_path=chunk.page_image_path,
                    )
                )

            else:
                payload = original_result.payload

                context_items.append(
                    ContextItem(
                        chunk_id=payload["chunk_id"],
                        document_name=payload["document_name"],
                        page_number=payload["page_number"],
                        content_type=payload["content_type"],
                        text=payload["text"],
                        image_path=payload.get("image_path"),
                        page_image_path=payload.get("page_image_path"),
                    )
                )

        return context_items

    def format_for_llm(
        self,
        context_items: list[ContextItem],
    ) -> str:
        """Format retrieved evidence for the LLM."""

        sections: list[str] = []

        for index, item in enumerate(context_items, start=1):
            section = (
                f"[SOURCE {index}]\n"
                f"Document: {item.document_name}\n"
                f"Page: {item.page_number}\n"
                f"Content Type: {item.content_type}\n"
                f"Chunk ID: {item.chunk_id}\n"
                f"Content:\n{item.text}"
            )

            sections.append(section)

        return "\n\n".join(sections)