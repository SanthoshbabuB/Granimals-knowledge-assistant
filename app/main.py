import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.memory.conversation import ConversationMemory
from app.rag.pipeline import RAGPipeline


settings = get_settings()

configure_logging(
    settings.log_level
)

logger = logging.getLogger(__name__)


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


rag_pipeline = RAGPipeline()

conversation_memory = ConversationMemory()


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
    )

    conversation_id: str | None = None


@app.post("/chat")
def chat(
    request: ChatRequest,
) -> dict:

    request_id = str(
        uuid.uuid4()
    )

    start_time = time.perf_counter()

    logger.info(
        "Chat request started",
        extra={
            "event": "chat_request_started",
            "request_id": request_id,
            "conversation_id": request.conversation_id,
        },
    )

    try:
        conversation_id = (
            request.conversation_id
            or str(uuid.uuid4())
        )

        history = (
            conversation_memory.get_messages(
                conversation_id
            )
        )

        result = rag_pipeline.answer(
            request.question,
            conversation_history=history,
        )

        conversation_memory.add_user_message(
            conversation_id,
            request.question,
        )

        conversation_memory.add_ai_message(
            conversation_id,
            result["answer"],
        )

        latency_ms = round(
            (
                time.perf_counter()
                - start_time
            )
            * 1000,
            2,
        )

        logger.info(
            "Chat request completed",
            extra={
                "event": "chat_request_completed",
                "request_id": request_id,
                "conversation_id": conversation_id,
                "latency_ms": latency_ms,
                "confidence": result.get(
                    "confidence"
                ),
                "retrieved_chunks": result.get(
                    "retrieved_chunks"
                ),
            },
        )

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"],
            "retrieved_chunks": result[
                "retrieved_chunks"
            ],
            "conversation_id": conversation_id,
        }

    except Exception as exc:

        latency_ms = round(
            (
                time.perf_counter()
                - start_time
            )
            * 1000,
            2,
        )

        logger.exception(
            "RAG request failed",
            extra={
                "event": "chat_request_failed",
                "request_id": request_id,
                "latency_ms": latency_ms,
            },
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
) -> dict:

    try:

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="File name is required.",
            )

        if not file.filename.lower().endswith(
            ".pdf"
        ):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported.",
            )

        documents_dir = Path(
            "data/documents"
        )

        documents_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_id = str(
            uuid.uuid4()
        )

        safe_filename = Path(
            file.filename
        ).name

        stored_filename = (
            f"{file_id}_{safe_filename}"
        )

        pdf_path = (
            documents_dir
            / stored_filename
        )

        file_content = (
            await file.read()
        )

        pdf_path.write_bytes(
            file_content
        )

        logger.info(
            "Uploaded document: %s",
            pdf_path,
        )

        result = (
            rag_pipeline.ingest_document(
                pdf_path
            )
        )

        return {
            "message": (
                "Document uploaded and "
                "indexed successfully."
            ),
            "filename": safe_filename,
            "path": str(pdf_path),
            "result": result,
        }

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "Document upload failed: %s",
            exc,
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc