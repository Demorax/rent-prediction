import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xgboost as xgb

from src.config import MODELS_DIR, XGBOOST_MODEL_PATH

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    predicted_rent: float
    shap_values: np.ndarray | None = None
    feature_contributions: dict | None = None


class RentPredictor:
    def __init__(self, model: xgb.XGBRegressor, preprocessor, target_encoder, feature_names: list[str]):
        self.model = model
        self.preprocessor = preprocessor
        self.target_encoder = target_encoder
        self.feature_names = feature_names

    @classmethod
    def from_pretrained(cls, model_path: Path | str | None = None, artifacts_path: Path | str | None = None):
        from src.models.train import load_model, load_artifacts

        model = load_model(model_path)
        artifacts = load_artifacts(artifacts_path)

        return cls(
            model=model,
            preprocessor=artifacts["preprocessor"],
            target_encoder=artifacts["target_encoder"],
            feature_names=artifacts["feature_names"],
        )

    def predict(self, data: dict, return_shap: bool = False, top_n: int = 5) -> PredictionResult:
        from src.data.preprocessing import preprocess_single_sample

        X = preprocess_single_sample(data, self.target_encoder, self.preprocessor)
        prediction = self.model.predict(X)[0]

        shap_values = None
        feature_contributions = None

        if return_shap:
            shap_values, feature_contributions = self._compute_shap(X, top_n)

        return PredictionResult(
            predicted_rent=float(prediction),
            shap_values=shap_values,
            feature_contributions=feature_contributions,
        )

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def _compute_shap(self, X: np.ndarray, top_n: int) -> tuple[np.ndarray, dict]:
        dmatrix = xgb.DMatrix(X)
        shap_raw = self.model.get_booster().predict(dmatrix, pred_contribs=True)

        shap_values = shap_raw[0, :-1]
        base_value = shap_raw[0, -1]

        contributions = {"base_value": float(base_value), "features": {}}
        for name, value in zip(self.feature_names, shap_values):
            contributions["features"][name] = float(value)

        sorted_features = sorted(contributions["features"].items(), key=lambda x: abs(x[1]), reverse=True)
        contributions["top_features"] = sorted_features[:top_n]

        return shap_values, contributions


def predict_rent(data: dict, model=None, preprocessor=None, target_encoder=None) -> float:
    if model is None:
        predictor = RentPredictor.from_pretrained()
        return predictor.predict(data).predicted_rent

    from src.data.preprocessing import preprocess_single_sample
    X = preprocess_single_sample(data, target_encoder, preprocessor)
    return float(model.predict(X)[0])
