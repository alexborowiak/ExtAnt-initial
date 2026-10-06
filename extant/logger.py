"""Logging configuration for interactive notebook sessions."""

import logging
import sys


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Log `name`'s records at `level` to stdout, leaving the root logger at WARNING.

    Replaces any existing root handlers, so it is safe to re-run in a notebook.
    Returns the configured logger.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s %(levelname)-7s %(name)s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger