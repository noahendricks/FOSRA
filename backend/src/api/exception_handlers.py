from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from loguru import logger


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred. Please try again.",
            "path": str(request.url.path),
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(Exception, generic_exception_handler)
    logger.info("Exception handlers registered successfully")
