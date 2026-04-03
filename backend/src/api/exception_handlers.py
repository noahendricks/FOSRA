from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi import status as http_status
from loguru import logger


async def validation_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    logger.bind(
        _structured={"path": str(request.url.path), "method": request.method}
    ).opt(exception=True).warning("Validation error")
    return JSONResponse(
        status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "message": str(exc),
            "path": str(request.url.path),
        },
    )


async def not_found_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=http_status.HTTP_404_NOT_FOUND,
        content={
            "error": "NotFound",
            "message": str(exc) or "Resource not found.",
            "path": str(request.url.path),
        },
    )


async def service_unavailable_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.bind(
        _structured={"path": str(request.url.path), "method": request.method}
    ).opt(exception=True).error("Service unavailable")
    return JSONResponse(
        status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "ServiceUnavailable",
            "message": "Service is temporarily unavailable. Please try again later.",
            "path": str(request.url.path),
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.bind(
        _structured={"path": str(request.url.path), "method": request.method}
    ).opt(exception=True).error("Unexpected error")
    return JSONResponse(
        status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred. Please try again.",
            "path": str(request.url.path),
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    from fastapi.exceptions import RequestValidationError
    from starlette.status import HTTP_404_NOT_FOUND, HTTP_503_SERVICE_UNAVAILABLE

    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(HTTP_404_NOT_FOUND, not_found_exception_handler)
    app.add_exception_handler(HTTP_503_SERVICE_UNAVAILABLE, service_unavailable_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
    logger.info("Exception handlers registered successfully")
