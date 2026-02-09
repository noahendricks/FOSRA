import os
import logging
import sys
from loguru import logger
from opentelemetry import _logs
from opentelemetry.sdk._logs import LoggerProvider


def patch_fosra_errors(record):
    exception = record.get("exception")
    if exception:
        exc_value = exception.value
        if hasattr(exc_value, "__dict__"):
            for key, value in exc_value.__dict__.items():
                if value is not None:
                    record["extra"][key] = value


class InterceptHandler(logging.Handler):
    """Bridge standard logging to Loguru."""

    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        # Escape angle brackets to prevent color tag parsing
        message: str = record.getMessage().replace("<", r"\<")

        logger.opt(depth=depth, exception=record.exc_info).log(level, message)


def setup_telemetry():
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"
    os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"

    if isinstance(_logs.get_logger_provider(), LoggerProvider):
        return

    logger.remove()

    logger.add(
        sys.stderr,
        level="TRACE",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    patched_logger = logger.patch(patch_fosra_errors)

    logging.basicConfig(
        handlers=[InterceptHandler()],
        level=2,
        force=True,
    )

    for name in ["uvicorn", "uvicorn.access", "fastapi"]:
        _log = logging.getLogger(name)
        _log.handlers = [InterceptHandler()]
        _log.propagate = False

    # 5. Instrumentation
    # logfire.instrument_pydantic(record="failure")
    # logfire.instrument_sqlalchemy()
    # logfire.instrument_httpx()

    return patched_logger


# def instrument_app(app):
#     logfire.instrument_fastapi(app)
