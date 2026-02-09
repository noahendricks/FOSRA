from loguru import logger
from taskiq import InMemoryBroker, TaskiqEvents, TaskiqState, TaskiqDepends
from backend.src.settings import settings
from backend.src.api.lifecycle import Infrastructure

broker = InMemoryBroker()


@broker.on_event(TaskiqEvents.WORKER_STARTUP)
async def startup(state: TaskiqState):
    """Initialize heavy singletons ONCE when the worker starts."""
    infra = Infrastructure(settings)

    # Store the infra instance in taskiq state
    state.infra = infra

    logger.success("Taskiq startup successful")


@broker.on_event(TaskiqEvents.WORKER_SHUTDOWN)
async def shutdown(state: TaskiqState):
    """Cleanup on worker shutdown."""
    logger.info("Taskiq worker shutting down...")

    if hasattr(state, "infra") and state.infra is not None:
        try:
            await state.infra.close()
            logger.info("Infrastructure closed successfully")
        except Exception as e:
            logger.error(f"Error closing infrastructure: {e}")
    else:
        logger.debug("No infrastructure to close")


def get_infra(state: TaskiqState = TaskiqDepends()) -> Infrastructure:
    return state.infra
