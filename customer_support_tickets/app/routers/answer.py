from time import perf_counter

from fastapi import APIRouter

from app.schemas.schema_answer import AnswerRequest, AnswerResponse, GeneratedAnswer
from app.services.evaluation_service import (
    calculate_rag_confidence_percent,
    evaluate_answer_pair_with_llm_metrics,
    evaluate_confidence,
    estimate_llm_cost_usd,
    estimate_llm_tokens,
)
from app.services.llm_service import generate_plain_answer
from app.services.rag_service import generate_rag_answer, retrieve_similar_tickets


router = APIRouter(prefix="/answer", tags=["answer"])


@router.post("/", response_model=AnswerResponse)
def answer_ticket(request: AnswerRequest):
    retrieved_tickets = retrieve_similar_tickets(request.message, top_k=request.top_k)
    rag_answer_text = generate_rag_answer(request.message, retrieved_tickets)

    llm_start = perf_counter()
    plain_answer_data = generate_plain_answer(request.message)
    llm_latency_ms = (perf_counter() - llm_start) * 1000

    llm_estimated_tokens = estimate_llm_tokens(
        request.message,
        str(plain_answer_data["text"]),
        str(plain_answer_data["confidence_basis"]),
    )
    llm_estimated_cost_usd = estimate_llm_cost_usd(llm_estimated_tokens)

    rag_answer = GeneratedAnswer(
        text=rag_answer_text,
        confidence_percent=calculate_rag_confidence_percent(retrieved_tickets),
        confidence_basis="Heuristic based on top retrieval match and average retrieval quality.",
    )
    plain_answer = GeneratedAnswer(
        text=str(plain_answer_data["text"]),
        confidence_percent=float(plain_answer_data["confidence_percent"]),
        confidence_basis=str(plain_answer_data["confidence_basis"]),
    )
    confidence = evaluate_confidence(rag_answer, plain_answer)

    return AnswerResponse(
        message=request.message,
        rag_answer=rag_answer,
        plain_llm_answer=plain_answer,
        retrieved_sources=retrieved_tickets,
        evaluation=evaluate_answer_pair_with_llm_metrics(
            confidence=confidence,
            llm_latency_ms=llm_latency_ms,
            llm_estimated_tokens=llm_estimated_tokens,
            llm_estimated_cost_usd=llm_estimated_cost_usd,
        ),
    )
