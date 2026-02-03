from fastapi import Request, status
from fastapi.responses import JSONResponse


class APIError(Exception):
    def __init__(self, message: str, status_code: int = 500, detail: str | None = None):
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(message)


class ModelNotLoadedError(APIError):
    def __init__(self):
        super().__init__(
            message="Model not loaded",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The prediction model is not available",
        )


class PredictionError(APIError):
    def __init__(self, detail: str | None = None):
        super().__init__(
            message="Prediction failed",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        )


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "detail": exc.detail},
    )
