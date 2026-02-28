"""Reusable QThread worker for running API calls off the main Qt thread."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QThread, Signal


class ApiWorker(QThread):
    """Execute *func(\*args, \*\*kwargs)* in a background thread.

    Signals:
        finished(result: object)  — emitted on success with the return value.
        failed(error: str)        — emitted when *func* raises an exception.
    """

    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        func: Callable[..., Any],
        *args: Any,
        parent: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent)
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:  # noqa: D401
        try:
            result = self._func(*self._args, **self._kwargs)
            self.finished.emit(result)
        except Exception as exc:
            from app.core.api_client import format_api_error

            self.failed.emit(format_api_error(exc))
