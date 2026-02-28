"""Centralised logging setup."""

import logging


def setup_logging(level: str) -> None:
    """Configure root logger with a simple format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
