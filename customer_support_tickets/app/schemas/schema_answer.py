from pydantic import BaseModel

from app.schemas.schema_retrieve import RetrievedTicket


class AnswerRequest(BaseModel):
    message: str
    top_k: int = 5


class GeneratedAnswer(BaseModel):
    text: str
    confidence_percent: float
    confidence_basis: str


class MetricComparison(BaseModel):
    metric_name: str
    rag_value: float
    plain_value: float
    unit: str
    preferred_answer: str
    explanation: str


class EvaluationInfo(BaseModel):
    evaluation_type: str
    preferred_answer: str
    confidence: MetricComparison
    llm_latency_ms: float
    llm_estimated_tokens: int
    llm_estimated_cost_usd: float
    summary: str


class AnswerResponse(BaseModel):
    message: str
    rag_answer: GeneratedAnswer
    plain_llm_answer: GeneratedAnswer
    retrieved_sources: list[RetrievedTicket]
    evaluation: EvaluationInfo
