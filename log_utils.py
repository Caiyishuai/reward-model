"""Centralised logging configuration for the AutoRM project.

Every module uses ``logger = logging.getLogger(__name__)``.
Entry-point scripts call :func:`setup_logging` once at startup.
"""

import logging
import os
import sys
from pathlib import Path

_LOG_FMT = "%(asctime)s %(levelname)-5s [%(name)s] %(message)s"
_DATE_FMT = "%H:%M:%S"
_configured = False


def setup_logging(
    level: int = logging.INFO,
    *,
    log_file: str | Path | None = None,
    rank: int | None = None,
) -> None:
    """Configure the root logger for the whole process.

    Args:
        level: Logging level for the main (rank-0) process.
        log_file: Optional path; a ``FileHandler`` is added when provided.
        rank: DDP local rank.  Non-zero ranks are silenced to WARNING.
    """
    global _configured  # noqa: PLW0603
    if _configured:
        return
    _configured = True

    if rank is None:
        rank = int(os.environ.get("LOCAL_RANK", 0))

    effective_level = level if rank == 0 else logging.WARNING

    root = logging.getLogger()
    root.setLevel(effective_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT))
    root.addHandler(handler)

    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT))
        fh.setLevel(logging.DEBUG)
        root.addHandler(fh)
