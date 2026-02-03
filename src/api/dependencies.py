from typing import Annotated

from fastapi import Depends
from loguru import logger

from src.api.exceptions import ModelNotLoadedError
from src.models.predict import RentPredictor


class PredictorService:
    _instance: RentPredictor | None = None
    _is_loaded: bool = False

    @classmethod
    def load(cls) -> None:
        if cls._instance is not None:
            return

        try:
            logger.info("Loading prediction model...")
            cls._instance = RentPredictor.from_pretrained()
            cls._is_loaded = True
            logger.info("Model loaded successfully")
        except FileNotFoundError as e:
            logger.warning(f"Model not found: {e}")
            cls._is_loaded = False

    @classmethod
    def get_predictor(cls) -> RentPredictor:
        if cls._instance is None:
            raise ModelNotLoadedError()
        return cls._instance

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._is_loaded


def get_predictor() -> RentPredictor:
    return PredictorService.get_predictor()


PredictorDep = Annotated[RentPredictor, Depends(get_predictor)]
