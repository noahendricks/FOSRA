import io
import logging
import sys
from typing import Any

from loguru import logger
from rich.console import Console
from rich.pretty import Pretty


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except (ValueError, KeyError):
            level = record.levelno if hasattr(record, "levelno") else "DEBUG"
        frame, depth = logging.currentframe(), 2
        message: str = record.getMessage().replace("<", r"\<")
        logger.opt(depth=depth, exception=record.exc_info).log(level, message)


def _pretty_render(obj: object) -> str:
    """Render any object using rich.pretty and return as a plain string with ANSI codes."""
    buf = io.StringIO()
    console = Console(file=buf, highlight=True, no_color=False, width=120)
    console.print(Pretty(obj, indent_size=1))
    return buf.getvalue().rstrip("\n")


LOGURU_FORMAT = "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <7}</level>|<cyan>{extra[short_name]}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>{extra[_rich_data]}\n"


def _format_exception(exc_info: tuple[Any, Any, Any]) -> str:
    """Format exception info (type, value, traceback) into a rich string."""
    if not exc_info or exc_info == (None, None, None):
        return ""

    exc_type, exc_val, exc_tb = exc_info
    parts = []

    if exc_type is not None:
        parts.append(f"\033[1m{exc_type.__name__}\033[0m")

    if exc_val is not None:
        parts.append(f"\033[91m{str(exc_val)}\033[0m")

    if exc_tb is not None:
        tb_lines = []
        while exc_tb is not None:
            frame = exc_tb.tb_frame
            filename = frame.f_code.co_filename.split("/")[-1]
            lineno = exc_tb.tb_lineno
            funcname = frame.f_code.co_name
            tb_lines.append(
                f"  File \033[2m{filename}:{lineno}\033[0m in \033[2m{funcname}\033[0m"
            )
            exc_tb = exc_tb.tb_next

        if tb_lines:
            parts.append("\n" + "\n".join(tb_lines))

    return " | ".join(parts)


def _process_record(record: dict[str, Any]) -> bool:
    extras = record.get("extra", {})
    parts = record["name"].split(".")
    extras["short_name"] = ".".join(parts[-2:]) if len(parts) >= 2 else record["name"]

    # Extract and pretty-print exception info if present
    exception_info = record.get("exception")
    exception_str = ""
    if exception_info and exception_info != (None, None, None):
        exception_str = _format_exception(exception_info)

    # patch fosra errors BEFORE rich rendering so exception fields get captured

    # handle _structured extra for rich.pretty rendering of named fields
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
            extra_lines = "\n" + "\n".join(rendered_parts)
            extras["_rich_data"] = (
                (exception_str + extra_lines) if exception_str else extra_lines
            )
        else:
            extras["_rich_data"] = exception_str if exception_str else ""
    else:
        # original behavior for general extras
        all_extras = {k: v for k, v in extras.items() if not k.startswith("_")}
        rich_parts = {}
        for key, val in all_extras.items():
            if isinstance(val, (dict, list, set, tuple)) or (
                hasattr(val, "__dict__") and not callable(val)
            ):
                rich_parts[key] = _pretty_render(val)
        if rich_parts or exception_str:
            rendered = "\n".join(
                f"  \033[2m{k}\033[0m = {v}" for k, v in rich_parts.items()
            )
            if exception_str:
                extras["_rich_data"] = exception_str + (
                    "\n" + rendered if rendered else ""
                )
            else:
                extras["_rich_data"] = "\n" + rendered if rendered else ""
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

    _ = logger.add(
        sink=sys.stderr,
        level="DEBUG",
        format=LOGURU_FORMAT,
        colorize=True,
        filter=_process_record,
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
