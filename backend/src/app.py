import asyncio
import os
import warnings
from asyncio import CancelledError
from contextlib import asynccontextmanager

import taskiq_fastapi
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.src.api.exception_handlers import register_exception_handlers
from backend.src.api.lifecycle import global_infra
from backend.src.api.routes.ingestion import router as ingestion_router
from backend.src.api.routes.ingestion_status import router as ingestion_status_router
from backend.src.api.routes.oc.state import log_process_start
from backend.src.api.routes.retrieval import router as retrieval_router
from backend.src.api.routes.tui import router as tui_router
from backend.src.services.session.langgraph_server import router as langgraph_router
from backend.src.logging_config import setup_logging
from backend.src.settings import settings
from backend.src.settings.observe import setup_telemetry
from backend.src.tasks.broker import broker

# install(show_locals=True)


# logfire.configure(
#     service_name="FOSRA",
#     send_to_logfire=False,
# )
#

taskiq_fastapi.init(broker, "backend.src.app:app")

warnings.filterwarnings("ignore", category=DeprecationWarning, module="litellm")
warnings.filterwarnings("ignore", category=ResourceWarning)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Starting application...")

        setup_logging()
        log_process_start()
        setup_telemetry()

        global_infra.init()

        await global_infra.init_models()

        from backend.src.api.schemas.provider_registry import _init_models_dev

        await _init_models_dev()

        app.state.infra = global_infra

        if not broker.is_worker_process:
            await broker.startup()
            logger.info("✓ Taskiq broker started")

        logger.info("✓ Application startup complete")

        import litellm

        litellm.suppress_debug_info = True

        yield

        async def print_routes():
            for route in app.routes:
                logger.debug(
                    "Route: {} | Name: {} | Methods:", route.url_path_for, route
                )

    except CancelledError:
        pass

    finally:
        logger.info("Shutting down application...")

        from backend.src.services.session.event_emitter import get_event_emitter

        await get_event_emitter().emit_server_instance_disposed(str(os.getcwd()))
        await asyncio.sleep(0.5)

        if not broker.is_worker_process:
            await broker.shutdown()
            logger.info("✓ Taskiq broker stopped")

        logger.info("✓ Logfire shutdown")

        await global_infra.close()
        logger.info("✓ Infrastructure closed")

        logger.info("✓ Application shutdown complete")


app = FastAPI(lifespan=lifespan)

taskiq_fastapi.populate_dependency_context(broker, app)

register_exception_handlers(app=app)

app.include_router(ingestion_router)
app.include_router(ingestion_status_router)
app.include_router(retrieval_router)
app.include_router(tui_router)
app.include_router(langgraph_router)


app.add_middleware(
    middleware_class=CORSMiddleware,
    allow_origins=settings.cors.allowed_origins,
    allow_methods=settings.cors.allowed_methods,
    allow_headers=settings.cors.allowed_headers,
)


# from rich.traceback import Traceback
#
# console = Console()
#
#
# @app.exception_handler(Exception)
# async def rich_exception_handler(request, exc):
#     console.print(
#         Traceback.from_exception(
#             type(exc),
#             exc,
#             exc.__traceback__,
#             show_locals=True,
#         )
#     )
#     return JSONResponse(status_code=500, content={"detail": str(exc)})


# lightweight liveness probe for k8s / load balancers
@app.get("/health")
async def health_check():
    return {"status": "ok", "ts": __import__("datetime").datetime.utcnow().isoformat()}


def run() -> None:
    """Entry point for `uv run fosra`."""
    import uvicorn
    import uvloop

    uvloop.install()
    uvicorn.run(
        "backend.src.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["./backend"],
        reload_delay=0.25,
        log_config=None,
    )


if __name__ == "__main__":
    run()
