from dataclasses import dataclass
import os
from pathlib import Path


def _load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        cleaned_value = value.strip()
        if not cleaned_value.startswith(("'", '"')) and " #" in cleaned_value:
            cleaned_value = cleaned_value.split(" #", 1)[0].strip()

        os.environ.setdefault(key.strip(), cleaned_value.strip('"').strip("'"))


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


_load_env_file()


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Customer Support Tickets API")
    dataset_path: str = os.getenv("DATASET_PATH", "twitter data/customer_support.csv")
    rag_sample_size: int = _get_int("RAG_SAMPLE_SIZE", 300_000)
    ingestion_batch_size: int = _get_int("INGESTION_BATCH_SIZE", 500)
    validation_sample_size: int = _get_int("VALIDATION_SAMPLE_SIZE", 2_000)
    ml_model_path: str = os.getenv("ML_MODEL_PATH", "models/priority_model.pkl")
    ml_model_accuracy_percent: float = float(os.getenv("ML_MODEL_ACCURACY_PERCENT", "97.0"))
    chroma_path: str = os.getenv("CHROMA_PATH", "rag/chroma_db")
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "customer_support_tickets")

    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_batch_size: int = _get_int("EMBEDDING_BATCH_SIZE", 100)


settings = Settings()
