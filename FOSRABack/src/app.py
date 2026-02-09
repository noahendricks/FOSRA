from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from loguru import logger
from FOSRABack.src.api.lifecycle import global_infra
from FOSRABack.src.tasks.broker import broker
from FOSRABack.src.settings.observe import setup_telemetry
import taskiq_fastapi

from FOSRABack.src.api.routes.user import router as user_router
from FOSRABack.src.api.routes.workspace import router as workspace_router
from FOSRABack.src.api.routes.test_routes import router as test_router
from FOSRABack.src.api.routes.llm import router as stream_router
from FOSRABack.src.api.routes.config import router as config_router
import warnings

from rich.traceback import install

install(show_locals=True)

from asyncio import CancelledError


from FOSRABack.src.api.exception_handlers import register_exception_handlers


# logfire.configure(
#     service_name="FOSRA",
#     send_to_logfire=False,
# )
#

taskiq_fastapi.init(broker, "FOSRABack.src.main:app")

warnings.filterwarnings("ignore", category=DeprecationWarning, module="litellm")
warnings.filterwarnings("ignore", category=ResourceWarning)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Starting application...")

        global_infra.init()

        setup_telemetry()

        await global_infra.init_models()

        app.state.infra = global_infra

        if not broker.is_worker_process:
            await broker.startup()
            logger.info("✓ Taskiq broker started")

        logger.info("✓ Application startup complete")

        import litellm

        litellm.set_verbose = False
        litellm.suppress_debug_info

        yield

        async def print_routes():
            for route in app.routes:
                print(f"Path: {route.url_path_for} | Name: {route} | Methods:")

    except CancelledError:
        pass

    finally:
        logger.info("Shutting down application...")

        if not broker.is_worker_process:
            await broker.shutdown()
            logger.info("✓ Taskiq broker stopped")

        logger.info("✓ Logfire shutdown")

        await global_infra.close()
        logger.info("✓ Infrastructure closed")

        logger.info("✓ Application shutdown complete")


app = FastAPI(lifespan=lifespan)


register_exception_handlers(app=app)

app.include_router(user_router)
app.include_router(workspace_router)
app.include_router(test_router)
app.include_router(stream_router)
app.include_router(config_router)


app.add_middleware(
    middleware_class=CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


from rich.console import Console
from rich.traceback import Traceback

console = Console()


@app.exception_handler(Exception)
async def rich_exception_handler(request, exc):
    console.print(
        Traceback.from_exception(
            type(exc),
            exc,
            exc.__traceback__,
            show_locals=True,
        )
    )
    return JSONResponse(status_code=500, content={"detail": str(exc)})


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
