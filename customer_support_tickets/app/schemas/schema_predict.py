from pydantic import BaseModel


class PredictRequest(BaseModel):
    message: str


class MlPrediction(BaseModel):
    priority: str
    model_accuracy_percent: float
    accuracy_basis: str


class LlmPrediction(BaseModel):
    priority: str
    confidence_percent: float
    confidence_basis: str
    latency_ms: float
    estimated_tokens: int
    estimated_cost_usd: float


class PredictEvaluation(BaseModel):
    evaluation_type: str
    preferred_method: str
    summary: str


class PredictResponse(BaseModel):
    message: str
    ml_prediction: MlPrediction
    llm_prediction: LlmPrediction
    evaluation: PredictEvaluation
