"""Stats tab — view index statistics and rebuild the HNSW index."""

from __future__ import annotations

import json
import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.api_client import ApiClient
from app.core.worker import ApiWorker


class StatsTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.api = ApiClient()
        self._worker: ApiWorker | None = None
        self._op_start: float = 0.0
        self.stats_view = QTextEdit()
        self.stats_view.setReadOnly(True)
        self.backend_label = QLabel("Embedding backend: -")
        self.last_rebuild_params: dict[str, int] | None = None
        self.last_rebuild_label = QLabel("Last rebuild params: -")
        self.latency_label = QLabel("Latency: - ms")

        self.m_input = QLineEdit("32")
        self.efc_input = QLineEdit("200")
        self.efs_input = QLineEdit("64")

        self.refresh_btn = QPushButton("Refresh stats")
        self.refresh_btn.clicked.connect(self._refresh)

        self.rebuild_btn = QPushButton("Rebuild HNSW")
        self.rebuild_btn.clicked.connect(self._rebuild)

        form = QFormLayout()
        form.addRow(QLabel("HNSW M"), self.m_input)
        form.addRow(QLabel("HNSW efConstruction"), self.efc_input)
        form.addRow(QLabel("HNSW efSearch"), self.efs_input)

        layout = QVBoxLayout()
        layout.addWidget(self.refresh_btn, alignment=Qt.AlignLeft)
        layout.addWidget(self.latency_label, alignment=Qt.AlignLeft)
        layout.addWidget(self.backend_label, alignment=Qt.AlignLeft)
        layout.addWidget(self.stats_view)
        layout.addWidget(self.last_rebuild_label)
        layout.addWidget(QLabel("Rebuild index (HNSW)"))
        layout.addLayout(form)
        layout.addWidget(self.rebuild_btn, alignment=Qt.AlignLeft)
        self.setLayout(layout)

    # ── helpers ───────────────────────────────────────────────────────────

    def _run(self, func, *args) -> None:
        self._op_start = time.perf_counter()
        self.refresh_btn.setEnabled(False)
        self.rebuild_btn.setEnabled(False)
        self._worker = ApiWorker(func, *args, parent=self)
        self._worker.finished.connect(self._on_success)
        self._worker.failed.connect(self._on_error)
        self._worker.start()

    def _on_success(self, result: object) -> None:
        self.refresh_btn.setEnabled(True)
        self.rebuild_btn.setEnabled(True)
        latency_ms = (time.perf_counter() - self._op_start) * 1000
        self.latency_label.setText(f"Latency: {latency_ms:.2f} ms")
        if isinstance(result, dict):
            backend = result.get("embedding_backend", "-")
            self.backend_label.setText(f"Embedding backend: {backend}")
        self.stats_view.setPlainText(json.dumps(result, indent=2, default=str))

    def _on_error(self, error: str) -> None:
        self.refresh_btn.setEnabled(True)
        self.rebuild_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", error)

    # ── actions ──────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        self._run(self.api.index_stats)

    def _rebuild(self) -> None:
        try:
            params = {
                "m": int(self.m_input.text()),
                "ef_construction": int(self.efc_input.text()),
                "ef_search": int(self.efs_input.text()),
            }
        except ValueError:
            QMessageBox.warning(self, "Invalid", "All HNSW params must be integers")
            return
        self.last_rebuild_params = params
        self.last_rebuild_label.setText(
            f"Last rebuild params: {json.dumps(self.last_rebuild_params)}"
        )
        self._run(self.api.rebuild_index, "hnsw", params)
