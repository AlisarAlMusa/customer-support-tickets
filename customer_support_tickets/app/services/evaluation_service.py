from app.schemas.schema_answer import EvaluationInfo, GeneratedAnswer, MetricComparison
from app.schemas.schema_predict import PredictEvaluation
from app.schemas.schema_retrieve import RetrievedTicket

HIGH_CONFIDENCE_DISTANCE = 0.1
MEDIUM_CONFIDENCE_DISTANCE = 0.6
LOW_CONFIDENCE_DISTANCE = 0.9
HIGH_CONFIDENCE_PERCENT = 90.0
MEDIUM_CONFIDENCE_PERCENT = 60.0
LOW_CONFIDENCE_PERCENT = 40.0
MIN_CONFIDENCE_DISTANCE = 1.0
MIN_CONFIDENCE_PERCENT = 30.0


def evaluate_confidence(rag_answer: GeneratedAnswer, plain_answer: GeneratedAnswer) -> MetricComparison:
    preferred_answer = "rag" if rag_answer.confidence_percent >= plain_answer.confidence_percent else "plain_llm"
    return MetricComparison(
        metric_name="confidence",
        rag_value=rag_answer.confidence_percent,
        plain_value=plain_answer.confidence_percent,
        unit="percent",
        preferred_answer=preferred_answer,
        explanation="Higher confidence percent is better.",
    )


def calculate_rag_confidence_percent(retrieved_tickets: list[RetrievedTicket]) -> float:
    if not retrieved_tickets:
        return 0.0

    top_distance = _ticket_distance(retrieved_tickets[0])
    avg_distance = sum(_ticket_distance(ticket) for ticket in retrieved_tickets) / len(retrieved_tickets)
    weighted_distance = (0.7 * top_distance) + (0.3 * avg_distance)
    return distance_to_confidence(weighted_distance)


def distance_to_confidence(distance: float) -> float:
    clamped_distance = max(0.0, distance)

    if clamped_distance <= HIGH_CONFIDENCE_DISTANCE:
        return HIGH_CONFIDENCE_PERCENT

    if clamped_distance <= MEDIUM_CONFIDENCE_DISTANCE:
        slope = (
            (MEDIUM_CONFIDENCE_PERCENT - HIGH_CONFIDENCE_PERCENT)
            / (MEDIUM_CONFIDENCE_DISTANCE - HIGH_CONFIDENCE_DISTANCE)
        )
        confidence = HIGH_CONFIDENCE_PERCENT + (
            (clamped_distance - HIGH_CONFIDENCE_DISTANCE) * slope
        )
        return round(max(0.0, min(100.0, confidence)), 1)

    if clamped_distance <= LOW_CONFIDENCE_DISTANCE:
        slope = (
            (LOW_CONFIDENCE_PERCENT - MEDIUM_CONFIDENCE_PERCENT)
            / (LOW_CONFIDENCE_DISTANCE - MEDIUM_CONFIDENCE_DISTANCE)
        )
        confidence = MEDIUM_CONFIDENCE_PERCENT + (
            (clamped_distance - MEDIUM_CONFIDENCE_DISTANCE) * slope
        )
        return round(max(0.0, min(100.0, confidence)), 1)

    if clamped_distance >= MIN_CONFIDENCE_DISTANCE:
        return MIN_CONFIDENCE_PERCENT

    slope = (
        (MIN_CONFIDENCE_PERCENT - LOW_CONFIDENCE_PERCENT)
        / (MIN_CONFIDENCE_DISTANCE - LOW_CONFIDENCE_DISTANCE)
    )
    confidence = LOW_CONFIDENCE_PERCENT + (
        (clamped_distance - LOW_CONFIDENCE_DISTANCE) * slope
    )
    return round(max(0.0, min(100.0, confidence)), 1)


def _ticket_distance(ticket: RetrievedTicket) -> float:
    if ticket.distance is not None:
        return ticket.distance
    return max(0.0, 1.0 - ticket.score)


def estimate_llm_tokens(*texts: str) -> int:
    return int(sum(max(1, len(text.split()) * 1.3) for text in texts))


def estimate_llm_cost_usd(estimated_tokens: int) -> float:
    return round(estimated_tokens * 0.000001, 6)


def evaluate_answer_pair(confidence: MetricComparison) -> EvaluationInfo:
    return evaluate_answer_pair_with_llm_metrics(
        confidence=confidence,
        llm_latency_ms=0.0,
        llm_estimated_tokens=0,
        llm_estimated_cost_usd=0.0,
    )


def evaluate_answer_pair_with_llm_metrics(
    confidence: MetricComparison,
    llm_latency_ms: float,
    llm_estimated_tokens: int,
    llm_estimated_cost_usd: float,
) -> EvaluationInfo:
    preferred_answer = confidence.preferred_answer
    summary = _hardcoded_evaluation_summary(
        confidence,
        llm_latency_ms=llm_latency_ms,
        llm_estimated_tokens=llm_estimated_tokens,
        llm_estimated_cost_usd=llm_estimated_cost_usd,
    )

    return EvaluationInfo(
        evaluation_type="tradeoff_summary",
        preferred_answer=preferred_answer,
        confidence=confidence,
        llm_latency_ms=round(llm_latency_ms, 1),
        llm_estimated_tokens=llm_estimated_tokens,
        llm_estimated_cost_usd=llm_estimated_cost_usd,
        summary=summary,
    )

# RAG vs non RAG analysis
def _hardcoded_evaluation_summary(
    confidence: MetricComparison,
    llm_latency_ms: float,
    llm_estimated_tokens: int,
    llm_estimated_cost_usd: float,
) -> str:
    gap = round(abs(confidence.rag_value - confidence.plain_value), 1)

    if confidence.preferred_answer == "rag":
        return (
            f"The RAG answer looks stronger on confidence ({confidence.rag_value}% vs "
            f"{confidence.plain_value}%). This means the retrieved examples gave "
            f"the answer specific and reliable grounding in past support behavior. "
            f"The plain LLM path still has measurable API overhead "
            f"({round(llm_latency_ms, 1)} ms, {llm_estimated_tokens} estimated tokens, "
            f"${llm_estimated_cost_usd:.6f}), while the RAG side is strong when retrieval is reliable on your own dataset. "
            f" The current gap is {gap} percentage points."
        )

    return (
        f"The plain LLM answer looks stronger on confidence ({confidence.plain_value}% vs "
        f"{confidence.rag_value}%). This means the model was able to answer "
        f"the issue directly without needing retrieved support history. The tradeoff is that "
        f"plain answers can sound fluent even when they are less grounded in historical cases, "
        f"and this path carries measurable API overhead ({round(llm_latency_ms, 1)} ms, "
        f"{llm_estimated_tokens} estimated tokens, ${llm_estimated_cost_usd:.6f}). "
        f"The current gap is {gap} percentage points."
    )


def evaluate_prediction_pair(
    ml_accuracy_percent: float,
    llm_confidence_percent: float,
    llm_latency_ms: float,
    llm_cost_usd: float,
) -> PredictEvaluation:
    preferred_method = "ml"
    summary = (
        f"The ML model is the stronger default for priority prediction in this project because "
        f"it is trained on your task-specific data and carries a tracked accuracy of "
        f"{ml_accuracy_percent}%. The LLM prediction is still useful as a flexible second opinion, "
        f"but it comes with latency ({round(llm_latency_ms, 1)} ms) and cost (${llm_cost_usd:.6f}) "
        f"while its confidence remains self-reported at {llm_confidence_percent}%. In short: ML is "
        f"better for stable operational prediction, and the LLM is better as an interpretable companion signal."
    )

    return PredictEvaluation(
        evaluation_type="hardcoded_tradeoff_summary",
        preferred_method=preferred_method,
        summary=summary,
    )
