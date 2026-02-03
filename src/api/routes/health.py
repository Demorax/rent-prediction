from fastapi import APIRouter

from src import __version__
from src.api.dependencies import PredictorService
from src.api.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    is_loaded = PredictorService.is_loaded()
    return HealthResponse(
        status="ok" if is_loaded else "degraded",
        model_loaded=is_loaded,
        version=__version__,
    )
