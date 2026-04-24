from functools import lru_cache
from pathlib import Path

import pandas as pd

from app.config import settings
from app.services.ml_compat import register_pickle_compat_classes


@lru_cache(maxsize=1)
def _load_model():
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("joblib is missing. Install requirements.txt first.") from exc

    model_path = Path(settings.ml_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"ML model file not found: {model_path}")

    register_pickle_compat_classes()
    model_package = joblib.load(model_path)
    if not isinstance(model_package, dict) or "pipeline" not in model_package:
        raise ValueError("Saved ML model file must contain a model package with a 'pipeline' key.")

    return model_package


def predict_priority(message: str) -> str:
    model_package = _load_model()
    pipeline = model_package["pipeline"]
    feature_columns = model_package.get("feature_columns", ["clean_text"])
    xgboost_outputs_numeric_labels = model_package.get("xgboost_outputs_numeric_labels", False)
    label_mapping = model_package.get("label_mapping", {})

    features = _build_feature_frame(message, feature_columns)
    prediction = pipeline.predict(features)[0]

    if xgboost_outputs_numeric_labels:
        prediction = label_mapping.get(int(prediction), prediction)

    return str(prediction)


def get_model_accuracy_percent() -> float:
    return settings.ml_model_accuracy_percent


def _build_feature_frame(message: str, feature_columns: list[str]) -> pd.DataFrame:
    row = {}
    for column in feature_columns:
        row[column] = message

    return pd.DataFrame([row], columns=feature_columns)
