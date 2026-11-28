import logging
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from copy import copy
from pathlib import Path


def exception_site(error: BaseException) -> str:
    """Identify the failing line without rendering exception messages or locals."""
    frame = error.__traceback__
    if frame is None:
        return "unknown"
    while frame.tb_next is not None:
        frame = frame.tb_next
    code = frame.tb_frame.f_code
    return f"{Path(code.co_filename).name}:{frame.tb_lineno} ({code.co_name})"


class FileLogFormatter(logging.Formatter):
    """Keep request query strings and exception details out of persisted logs."""

    def format(self, record: logging.LogRecord) -> str:
        record = copy(record)
        if record.name == "uvicorn.access" and isinstance(record.args, tuple) and len(record.args) == 5:
            client, method, path, version, status = record.args
            record.args = (client, method, path.partition("?")[0], version, status)
        # A console formatter may already have cached the full exception text.
        record.exc_text = None
        return super().format(record)

    def formatException(self, exc_info) -> str:
        exc_type, _, frames = exc_info
        stack = "".join(traceback.format_tb(frames))
        return f"{stack}{exc_type.__name__}"


@contextmanager
def log_to_file(path: Path, logger: logging.Logger, *other_loggers: logging.Logger) -> Iterator[None]:
    """Add a file destination without changing existing console handlers."""
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.touch(mode=0o600, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
    except OSError as error:
        logger.warning("Could not open log file %s (%s); continuing with console logging", path, type(error).__name__)
        yield
        return

    handler.setLevel(logging.INFO)
    handler.setFormatter(FileLogFormatter("%(asctime)s %(levelname)s %(name)s [%(process)d] %(message)s"))
    loggers = (logger, *other_loggers)
    for destination in loggers:
        destination.addHandler(handler)
    try:
        yield
    finally:
        for destination in loggers:
            destination.removeHandler(handler)
        handler.close()
