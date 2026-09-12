import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(
    logging.Formatter
):
    """Format logs as JSON."""

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:
        log_data = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "event"):
            log_data["event"] = record.event

        if hasattr(record, "request_id"):
            log_data["request_id"] = (
                record.request_id
            )

        if hasattr(record, "conversation_id"):
            log_data["conversation_id"] = (
                record.conversation_id
            )

        if hasattr(record, "latency_ms"):
            log_data["latency_ms"] = (
                record.latency_ms
            )

        if hasattr(record, "confidence"):
            log_data["confidence"] = (
                record.confidence
            )

        if hasattr(record, "retrieved_chunks"):
            log_data["retrieved_chunks"] = (
                record.retrieved_chunks
            )

        if record.exc_info:
            log_data["exception"] = self.formatException(
                record.exc_info
            )

        return json.dumps(
            log_data,
            ensure_ascii=False,
        )


def configure_logging(
    level: str = "INFO",
) -> None:
    """Configure application-wide structured logging."""

    handler = logging.StreamHandler(
        sys.stdout
    )

    handler.setFormatter(
        JsonFormatter()
    )

    root_logger = logging.getLogger()

    root_logger.handlers.clear()

    root_logger.setLevel(
        getattr(
            logging,
            level.upper(),
            logging.INFO,
        )
    )

    root_logger.addHandler(
        handler
    )