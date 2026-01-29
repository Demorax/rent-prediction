from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from src import __version__
from src.api.schemas import ApartmentInput, HealthResponse, PredictionResponse, PredictionWithShapResponse
from src.models.predict import RentPredictor

predictor: RentPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Loading model...")
    try:
        predictor = RentPredictor.from_pretrained()
        logger.info("Model loaded")
    except FileNotFoundError as e:
        logger.warning(f"Model not found: {e}")
    yield


app = FastAPI(title="Czech Rent Prediction API", version=__version__, lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if predictor else "degraded",
        model_loaded=predictor is not None,
        version=__version__,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(apartment: ApartmentInput):
    if not predictor:
        raise HTTPException(503, "Model not loaded")

    result = predictor.predict(apartment.model_dump())
    return PredictionResponse(predicted_rent=round(result.predicted_rent, 2))


@app.post("/predict/explain", response_model=PredictionWithShapResponse)
def predict_explain(apartment: ApartmentInput):
    if not predictor:
        raise HTTPException(503, "Model not loaded")

    result = predictor.predict(apartment.model_dump(), return_shap=True)
    return PredictionWithShapResponse(
        predicted_rent=round(result.predicted_rent, 2),
        shap_base_value=result.feature_contributions["base_value"],
        top_features=result.feature_contributions["top_features"],
    )
