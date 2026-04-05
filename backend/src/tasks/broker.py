from typing import Any
from loguru import logger
from taskiq import InMemoryBroker, TaskiqEvents, TaskiqState, TaskiqDepends
from taskiq.middlewares import SmartRetryMiddleware
from taskiq.abc.middleware import TaskiqMiddleware
from backend.src.settings import settings
from backend.src.api.lifecycle import Infrastructure

broker = InMemoryBroker()

broker.add_middlewares(
    SmartRetryMiddleware(
        default_retry_count=5,
        default_delay=10,
        use_jitter=True,
        use_delay_exponent=True,
        max_delay_exponent=120,
    )
)


class TaskiqObserverMiddleware(TaskiqMiddleware):
    async def pre_execute(self, message: Any) -> Any:  # type: ignore[type-arg]
        logger.debug("[taskiq] task queued: {}", message.task_name)
        return message

    async def on_error(  # type: ignore[type-arg]
        self, message: Any, result: Any, exception: BaseException
    ) -> None:
        logger.opt(exception=True).error("[taskiq] task failed: {}", message.task_name)

    async def post_execute(self, message: Any, result: Any) -> None:  # type: ignore[type-arg]
        logger.debug("[taskiq] task completed: {}", message.task_name)


broker.add_middlewares(TaskiqObserverMiddleware())


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
            logger.opt(exception=True).error("Error closing infrastructure")
    else:
        logger.debug("No infrastructure to close")


def get_infra(state: TaskiqState = TaskiqDepends()) -> Infrastructure:
    from backend.src.api.lifecycle import global_infra

    return (
        state.infra
        if hasattr(state, "infra") and state.infra is not None
        else global_infra
    )
