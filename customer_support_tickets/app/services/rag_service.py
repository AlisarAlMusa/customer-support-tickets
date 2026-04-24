from typing import Any

from app.schemas.schema_retrieve import RetrievedTicket
from rag.embedder import embed_query
from rag.store import get_or_create_collection


def retrieve_similar_tickets(query: str, top_k: int = 5) -> list[RetrievedTicket]:
    collection = get_or_create_collection()
    collection_count = collection.count()
    if collection_count == 0:
        return []

    query_embedding = embed_query(query)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection_count),
        include=["documents", "metadatas", "distances"],
    )

    documents = _first_or_empty(result.get("documents"))
    metadatas = _first_or_empty(result.get("metadatas"))
    distances = _first_or_empty(result.get("distances"))

    tickets: list[RetrievedTicket] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}
        tickets.append(
            RetrievedTicket(
                text=document or "",
                score=_distance_to_score(distance),
                response_text=metadata.get("response_text"),
                company=_safe_str(metadata.get("company")),
                source=_safe_str(metadata.get("source")),
                created_at=_safe_str(metadata.get("created_at")),
                customer_tweet_id=_safe_int(metadata.get("customer_tweet_id")),
                company_response_id=_safe_int(metadata.get("company_response_id")),
            )
        )

    return tickets


def generate_rag_answer(message: str, retrieved_tickets: list[RetrievedTicket]) -> str:
    from app.services.llm_service import generate_rag_grounded_answer

    context_blocks = []
    for idx, ticket in enumerate(retrieved_tickets, start=1):
        context_blocks.append(
            "\n".join(
                [
                    f"Example {idx}",
                    f"Customer issue: {ticket.text}",
                    f"Company response: {ticket.response_text or ''}",
                    f"Company: {ticket.company or ''}",
                    f"Similarity score: {ticket.score:.2f}",
                ]
            )
        )

    retrieved_context = "\n\n".join(context_blocks) if context_blocks else "No retrieved context."
    return generate_rag_grounded_answer(message, retrieved_context)


def _first_or_empty(value: Any) -> list[Any]:
    if not value:
        return []
    return value[0]


def _distance_to_score(distance: Any) -> float:
    try:
        numeric_distance = float(distance)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, 1.0 - numeric_distance), 4)


def _safe_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
