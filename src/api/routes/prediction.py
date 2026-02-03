from fastapi import APIRouter
from loguru import logger

from src.api.dependencies import PredictorDep
from src.api.exceptions import PredictionError
from src.api.schemas import (
    ApartmentInput,
    FeatureContribution,
    PredictionResponse,
    PredictionWithExplanationResponse,
)

router = APIRouter(prefix="/predict", tags=["Prediction"])


def _to_dict(apartment: ApartmentInput) -> dict:
    return {
        "city": apartment.city,
        "region": apartment.region,
        "floor_space": apartment.floor_space,
        "land_space": apartment.land_space,
        "disposition": apartment.disposition.value,
        "building_type": apartment.building_type.value,
        "condition": apartment.condition.value,
        "equipment": apartment.equipment.value,
        "penb": apartment.penb.value,
    }


@router.post("", response_model=PredictionResponse)
def predict(apartment: ApartmentInput, predictor: PredictorDep) -> PredictionResponse:
    try:
        result = predictor.predict(_to_dict(apartment))
        return PredictionResponse(predicted_rent=round(result.predicted_rent, 2))
    except Exception as e:
        logger.exception("Prediction failed")
        raise PredictionError(detail=str(e))


@router.post("/explain", response_model=PredictionWithExplanationResponse)
def predict_with_explanation(
    apartment: ApartmentInput, predictor: PredictorDep
) -> PredictionWithExplanationResponse:
    try:
        result = predictor.predict(_to_dict(apartment), return_shap=True)

        top_features = [
            FeatureContribution(feature=name, contribution=value)
            for name, value in result.feature_contributions["top_features"]
        ]

        return PredictionWithExplanationResponse(
            predicted_rent=round(result.predicted_rent, 2),
            base_value=result.feature_contributions["base_value"],
            top_features=top_features,
        )
    except Exception as e:
        logger.exception("Prediction failed")
        raise PredictionError(detail=str(e))
