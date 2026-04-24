from time import perf_counter

from fastapi import APIRouter

from app.logging_setup import log_audit_event
from app.schemas.schema_predict import (
    LlmPrediction,
    MlPrediction,
    PredictRequest,
    PredictResponse,
)
from app.services.evaluation_service import (
    estimate_llm_cost_usd,
    estimate_llm_tokens,
    evaluate_prediction_pair,
)
from app.services.llm_service import predict_priority_zero_shot
from app.services.ml_service import get_model_accuracy_percent, predict_priority


router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("/", response_model=PredictResponse)
def predict_ticket_priority(request: PredictRequest):
    log_audit_event("predict_requested", query=request.message)

    ml_prediction = MlPrediction(
        priority=predict_priority(request.message),
        model_accuracy_percent=get_model_accuracy_percent(),
        accuracy_basis="accuracy metric from the trained ML model.",
    )

    llm_start = perf_counter()
    llm_prediction_data = predict_priority_zero_shot(request.message)
    llm_latency_ms = (perf_counter() - llm_start) * 1000
    llm_estimated_tokens = estimate_llm_tokens(
        request.message,
        str(llm_prediction_data["priority"]),
        str(llm_prediction_data["confidence_basis"]),
    )
    llm_estimated_cost_usd = estimate_llm_cost_usd(llm_estimated_tokens)
    llm_prediction = LlmPrediction(
        priority=str(llm_prediction_data["priority"]),
        confidence_percent=float(llm_prediction_data["confidence_percent"]),
        confidence_basis=str(llm_prediction_data["confidence_basis"]),
        latency_ms=round(llm_latency_ms, 1),
        estimated_tokens=llm_estimated_tokens,
        estimated_cost_usd=llm_estimated_cost_usd,
    )

    response = PredictResponse(
        message=request.message,
        ml_prediction=ml_prediction,
        llm_prediction=llm_prediction,
        evaluation=evaluate_prediction_pair(
            ml_accuracy_percent=ml_prediction.model_accuracy_percent,
            llm_confidence_percent=llm_prediction.confidence_percent,
            llm_latency_ms=llm_prediction.latency_ms,
            llm_cost_usd=llm_prediction.estimated_cost_usd,
        ),
    )
    log_audit_event(
        "predict_completed",
        query=request.message,
        ml_prediction=response.ml_prediction.model_dump(),
        llm_prediction=response.llm_prediction.model_dump(),
        evaluation=response.evaluation.model_dump(),
    )
    return response
