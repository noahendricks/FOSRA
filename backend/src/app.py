import asyncio
import os
import warnings
from asyncio import CancelledError
from contextlib import asynccontextmanager

import taskiq_fastapi
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from rich.traceback import install

from backend.src.api.exception_handlers import register_exception_handlers
from backend.src.api.lifecycle import global_infra
from backend.src.api.routes.ingestion import router as ingestion_router
from backend.src.api.routes.tui import router as tui_router
from backend.src.api.routes.workspace import router as workspace_router
from backend.src.logging_config import setup_logging
from backend.src.api.routes.oc.state import log_process_start
from backend.src.settings.observe import setup_telemetry
from backend.src.tasks.broker import broker
from backend.src.settings import settings

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

app.include_router(workspace_router)
app.include_router(ingestion_router)
app.include_router(tui_router)


app.add_middleware(
    middleware_class=CORSMiddleware,
    allow_origins=settings.cors.allowed_origins,
    allow_methods=settings.cors.allowed_methods,
    allow_headers=settings.cors.allowed_headers,
)


from rich.console import Console

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


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["./"],
        reload_delay=0.25,
    )
