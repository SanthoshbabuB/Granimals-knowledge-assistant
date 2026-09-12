SYSTEM_PROMPT = """
You are a document question-answering assistant.

Your job is to answer the USER QUESTION using ONLY the SOURCE CONTEXT.

Rules:
- Answer the question directly.
- Use only information present in the SOURCE CONTEXT.
- Do not use outside knowledge.
- Do not guess or invent information.
- Do not repeat the SOURCE CONTEXT.
- Do not describe the retrieval process.
- Do not output document metadata unless it is necessary for the citation.
- Do not output labels such as "Document Name", "Page Number",
  "Content Type", "Chunk ID", or "Content".
- If the answer is not present in the SOURCE CONTEXT, say:
  "The information was not found in the provided documents."
- Keep the answer concise, normally 1-3 sentences.
- Include a citation for factual claims.

example:

question : What was Atlas Copco Group's revenue in 2024?

answer: "Atlas Copco Group's revenue in 2024 was MSEK 176,771. [AtlasCopca-annual-report-2024.pdf, Page 5]",
        "sources": [...],
        "confidence": 0.68,
        "retrieved_chunks": 5,
        "conversation_id": "..."

Citation format:
[Document Name, Page X]
"""


def build_prompt(
    question: str,
    context: str,
    conversation_history: list | None = None,
) -> str:

    history_text = ""

    if conversation_history:
        history_lines = []

        for message in conversation_history[-6:]:
            role = (
                "User"
                if message.type == "human"
                else "Assistant"
            )

            history_lines.append(
                f"{role}: {message.content}"
            )

        history_text = "\n".join(
            history_lines
        )

    return f"""
{SYSTEM_PROMPT}

CONVERSATION HISTORY:
{history_text}

USER QUESTION:
{question}

SOURCE CONTEXT:
{context}

IMPORTANT:
Use the SOURCE CONTEXT to answer the USER QUESTION.
Do not repeat the context.
Do not output the context structure or metadata.
Return only the final answer with a citation.

FINAL ANSWER:
"""