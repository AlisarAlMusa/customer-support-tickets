from typing import Any

from app.logging_setup import log_audit_event
from app.schemas.schema_retrieve import RetrievedTicket
from rag.embedder import embed_query
from rag.store import get_or_create_collection

WEAK_RETRIEVAL_THRESHOLD = 0.45


def retrieve_similar_tickets(query: str, top_k: int = 5) -> list[RetrievedTicket]:
    collection = get_or_create_collection()
    collection_count = collection.count()
    if collection_count == 0:
        log_audit_event(
            "retrieval_completed",
            query=query,
            top_k=top_k,
            collection_count=collection_count,
            retrieved_count=0,
            retrieved_tickets=[],
        )
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

    log_audit_event(
        "retrieval_completed",
        query=query,
        top_k=top_k,
        collection_count=collection_count,
        retrieved_count=len(tickets),
        retrieved_tickets=[
            {
                "rank": index,
                "text": ticket.text,
                "response_text": ticket.response_text,
                "company": ticket.company,
                "score": ticket.score,
                "created_at": ticket.created_at,
                "customer_tweet_id": ticket.customer_tweet_id,
                "company_response_id": ticket.company_response_id,
            }
            for index, ticket in enumerate(tickets, start=1)
        ],
    )

    return tickets


def generate_rag_answer(message: str, retrieved_tickets: list[RetrievedTicket]) -> str:
    from app.services.llm_service import generate_rag_grounded_answer

    if not retrieved_tickets:
        fallback = (
            "I'm not confident we have close enough past examples to answer this reliably right now."
        )
        log_audit_event(
            "rag_fallback_used",
            query=message,
            reason="no_retrieved_tickets",
            fallback=fallback,
        )
        return fallback

    top_score = retrieved_tickets[0].score
    if top_score < WEAK_RETRIEVAL_THRESHOLD:
        fallback = (
            "I'm not confident the retrieved examples are close enough to answer this reliably."
        )
        log_audit_event(
            "rag_fallback_used",
            query=message,
            reason="weak_retrieval",
            top_score=top_score,
            threshold=WEAK_RETRIEVAL_THRESHOLD,
            fallback=fallback,
        )
        return fallback

    context_blocks = []
    for idx, ticket in enumerate(retrieved_tickets, start=1):
        context_blocks.append(
            "\n".join(
                [
                    f"Example {idx}",
                    f"Customer issue: {ticket.text}",
                    f"Company response: {ticket.response_text or ''}",
                    f"Company: {ticket.company or ''}",
                    f"Similarity score (0 to 1, higher is better): {ticket.score:.2f}",
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
    return round(min(1.0, max(0.0, 1.0 - numeric_distance)), 4)


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
