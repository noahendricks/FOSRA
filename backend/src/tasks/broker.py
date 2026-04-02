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
    async def pre_execute(self, receiver, task) -> None:
        logger.debug(f"[taskiq] task queued: {task.task_name}")

    async def on_error(self, receiver, task, error) -> None:
        logger.error(f"[taskiq] task failed: {task.task_name} | error: {error}")

    async def post_execute(self, receiver, task) -> None:
        logger.debug(f"[taskiq] task completed: {task.task_name}")


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
            logger.error(f"Error closing infrastructure: {e}")
    else:
        logger.debug("No infrastructure to close")


def get_infra(state: TaskiqState = TaskiqDepends()) -> Infrastructure:
    from backend.src.api.lifecycle import global_infra

    if isinstance(state, TaskiqState):
        return state.infra
    return global_infra
