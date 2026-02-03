from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from src import __version__
from src.api.dependencies import PredictorService
from src.api.exceptions import APIError, api_error_handler
from src.api.routes import health_router, prediction_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting application...")
    PredictorService.load()
    yield
    logger.info("Shutting down application...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Czech Rent Prediction API",
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_exception_handler(APIError, api_error_handler)
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    app.include_router(health_router)
    app.include_router(prediction_router, prefix="/v1")

    return app


app = create_app()
