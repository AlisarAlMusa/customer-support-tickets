import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


LOG_DIR = Path("logs")
APP_LOG_PATH = LOG_DIR / "app.log"
AUDIT_LOG_PATH = LOG_DIR / "query_events.log"


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    if getattr(root_logger, "_customer_support_logging_configured", False):
        return

    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    app_file_handler = RotatingFileHandler(
        APP_LOG_PATH,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    app_file_handler.setLevel(logging.INFO)
    app_file_handler.setFormatter(formatter)
    root_logger.addHandler(app_file_handler)

    audit_logger = logging.getLogger("app.audit")
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False

    audit_file_handler = RotatingFileHandler(
        AUDIT_LOG_PATH,
        maxBytes=1_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    audit_file_handler.setLevel(logging.INFO)
    audit_file_handler.setFormatter(formatter)
    audit_logger.addHandler(audit_file_handler)

    root_logger._customer_support_logging_configured = True  # type: ignore[attr-defined]


def log_audit_event(event: str, **payload: Any) -> None:
    audit_logger = logging.getLogger("app.audit")
    message = json.dumps(
        {"event": event, **_sanitize_payload(payload)},
        ensure_ascii=True,
        default=str,
    )
    audit_logger.info(message)


def _sanitize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: _sanitize_value(value) for key, value in payload.items()}


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return _truncate_text(value)
    if isinstance(value, dict):
        return {str(key): _sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_value(item) for item in value]
    return value


def _truncate_text(value: str, limit: int = 1_000) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."
