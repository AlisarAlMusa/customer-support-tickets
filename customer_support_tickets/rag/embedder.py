from collections.abc import Iterable
from typing import Any

from app.config import settings
from app.services.openai_service import get_openai_client


def build_chunk_embedding_pairs(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    embeddings = embed_chunk_customer_texts(chunks)
    return [
        {"chunk": chunk, "embedding": embedding}
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]


def embed_chunk_customer_texts(chunks: list[dict[str, Any]]) -> list[list[float]]:
    texts = [chunk["text"] for chunk in chunks]
    return embed_texts(texts)


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]


def embed_texts(
    texts: list[str],
    model: str | None = None,
    batch_size: int | None = None,
) -> list[list[float]]:
    if not texts:
        return []
    if any(not text for text in texts):
        raise ValueError("Cannot embed empty text.")

    client = get_openai_client()
    embedding_model = model or settings.openai_embedding_model
    size = batch_size or settings.embedding_batch_size

    embeddings: list[list[float]] = []
    for batch in _batched(texts, size):
        response = client.embeddings.create(
            model=embedding_model,
            input=batch,
        )
        embeddings.extend(item.embedding for item in response.data)

    return embeddings


def _batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
