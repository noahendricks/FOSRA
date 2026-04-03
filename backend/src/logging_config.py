import io
import logging
import sys
from loguru import logger
from rich.console import Console
from rich.pretty import Pretty


class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except (ValueError, KeyError):
            level = record.levelno if hasattr(record, "levelno") else "DEBUG"
        frame, depth = logging.currentframe(), 2
        message: str = record.getMessage().replace("<", r"\<")
        logger.opt(depth=depth, exception=record.exc_info).log(level, message)


def patch_fosra_errors(record):
    exception = record.get("exception")
    if exception:
        exc_value = exception.value
        if hasattr(exc_value, "__dict__"):
            for key, value in exc_value.__dict__.items():
                if value is not None:
                    record["extra"][key] = value


def _pretty_render(obj: object) -> str:
    """Render any object using rich.pretty and return as a plain string with ANSI codes."""
    buf = io.StringIO()
    console = Console(file=buf, highlight=True, no_color=False, width=120)
    console.print(Pretty(obj, indent_size=2))
    return buf.getvalue().rstrip("\n")


LOGURU_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | "
    "<level>{level: <7}</level> | "
    "<cyan>{extra[short_name]}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
    "{extra[_rich_data]}"
    "\n"
)


def _process_record(record: dict) -> bool:
    extras = record.get("extra", {})
    parts = record["name"].split(".")
    extras["short_name"] = ".".join(parts[-2:]) if len(parts) >= 2 else record["name"]

    # Patch fosra errors BEFORE rich rendering so exception fields get captured
    patch_fosra_errors(record)

    # Handle _structured extra for rich.pretty rendering of named fields
    structured = extras.pop("_structured", None)
    if structured and isinstance(structured, dict):
        rendered_parts = []
        for key, val in structured.items():
            if isinstance(val, (dict, list, set, tuple)) or (
                hasattr(val, "__dict__") and not callable(val)
            ):
                rendered_parts.append(f"  \033[2m{key}\033[0m = {_pretty_render(val)}")
            else:
                rendered_parts.append(f"  \033[2m{key}\033[0m = {val!r}")
        if rendered_parts:
            extras["_rich_data"] = "\n" + "\n".join(rendered_parts)
        else:
            extras["_rich_data"] = ""
    else:
        # Original behavior for general extras
        all_extras = {k: v for k, v in extras.items() if not k.startswith("_")}
        rich_parts = {}
        for key, val in all_extras.items():
            if isinstance(val, (dict, list, set, tuple)) or (
                hasattr(val, "__dict__") and not callable(val)
            ):
                rich_parts[key] = _pretty_render(val)
        if rich_parts:
            rendered = "\n".join(
                f"  \033[2m{k}\033[0m = {v}" for k, v in rich_parts.items()
            )
            extras["_rich_data"] = "\n" + rendered
        else:
            extras["_rich_data"] = ""

    return True


def setup_logging():
    """
    Configure loguru with:
    - Rich.pretty rendering for dicts, lists, sets, tuples, and objects in .bind() extras
    - Short module name in log output
    - InterceptHandler for standard logging (uvicorn)
    - fosra error patching
    - stderr sink (colorized)
    """
    try:
        logger.remove()
    except Exception:
        pass

    logger.add(
        sys.stderr,
        format=LOGURU_FORMAT,
        filter=_process_record,
        level="DEBUG",
        colorize=True,
    )

    logging.basicConfig(
        handlers=[InterceptHandler()],
        level=2,
        force=True,
    )

    for name in ["uvicorn", "uvicorn.access", "fastapi"]:
        _log = logging.getLogger(name)
        _log.handlers = [InterceptHandler()]
        _log.propagate = False

    for name in ["litellm", "LiteLLM"]:
        _log = logging.getLogger(name)
        _log.setLevel(logging.WARNING)
        _log.handlers = []
        _log.propagate = False

    for name in ["httpcore", "httpx", "aiohttp"]:
        _log = logging.getLogger(name)
        _log.setLevel(logging.WARNING)
        _log.handlers = []
        _log.propagate = False
