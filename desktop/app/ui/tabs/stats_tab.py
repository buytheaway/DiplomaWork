from __future__ import annotations

import json
import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
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
from app.ui.widgets import Card, DimLabel, InfoRow, SectionHeading, StatBox


class StatsTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.api = ApiClient()
        self._worker: ApiWorker | None = None
        self._op_start: float = 0.0
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        # --- dashboard row ---
        dash_row = QHBoxLayout()
        dash_row.setSpacing(12)

        self.stat_backend = StatBox("Backend", "-")
        self.stat_vectors = StatBox("Vectors", "-")
        self.stat_dim = StatBox("Dim", "-")
        self.stat_index_type = StatBox("Index type", "-")

        for box in [self.stat_backend, self.stat_vectors, self.stat_dim, self.stat_index_type]:
            card = Card()
            card.body().addWidget(box)
            dash_row.addWidget(card, 1)

        root.addLayout(dash_row)

        # --- main area: raw stats + rebuild ---
        body_row = QHBoxLayout()
        body_row.setSpacing(16)

        # Raw stats
        stats_card = Card()
        sc = stats_card.body()

        stats_header = QHBoxLayout()
        stats_header.addWidget(SectionHeading("Index stats"))
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setFixedWidth(90)
        self.refresh_btn.clicked.connect(self._refresh)
        stats_header.addStretch()
        self.latency_info = DimLabel("")
        stats_header.addWidget(self.latency_info)
        stats_header.addWidget(self.refresh_btn)
        sc.addLayout(stats_header)

        self.stats_view = QTextEdit()
        self.stats_view.setReadOnly(True)
        self.stats_view.setPlaceholderText("Click Refresh to load index stats...")
        sc.addWidget(self.stats_view)

        body_row.addWidget(stats_card, 2)

        # Rebuild panel
        rebuild_card = Card()
        rc = rebuild_card.body()
        rc.addWidget(SectionHeading("Rebuild index"))
        rc.addWidget(DimLabel("Build an HNSW index from current embeddings"))
        rc.addSpacing(8)

        rc.addWidget(DimLabel("M"))
        self.m_input = QLineEdit("32")
        rc.addWidget(self.m_input)

        rc.addWidget(DimLabel("efConstruction"))
        self.efc_input = QLineEdit("200")
        rc.addWidget(self.efc_input)

        rc.addWidget(DimLabel("efSearch"))
        self.efs_input = QLineEdit("64")
        rc.addWidget(self.efs_input)

        rc.addSpacing(8)

        self.rebuild_btn = QPushButton("Rebuild HNSW")
        self.rebuild_btn.setObjectName("primary")
        self.rebuild_btn.clicked.connect(self._rebuild)
        rc.addWidget(self.rebuild_btn)

        self.rebuild_status = DimLabel("")
        rc.addWidget(self.rebuild_status)
        rc.addStretch()

        body_row.addWidget(rebuild_card, 1)

        root.addLayout(body_row, 1)

    # --- helpers ---

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
        self.latency_info.setText(f"{latency_ms:.0f} ms")

        if isinstance(result, dict):
            self.stat_backend.set_value(result.get("embedding_backend", "-"))
            self.stat_vectors.set_value(str(result.get("total_vectors", "-")))
            self.stat_dim.set_value(str(result.get("dim", "-")))
            self.stat_index_type.set_value(result.get("index_type", "-"))

        self.stats_view.setPlainText(json.dumps(result, indent=2, default=str))
        self.rebuild_status.setText("")

    def _on_error(self, error: str) -> None:
        self.refresh_btn.setEnabled(True)
        self.rebuild_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", error)

    # --- actions ---

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
        self.rebuild_status.setText("Rebuilding...")
        self._run(self.api.rebuild_index, "hnsw", params)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Автозагрузка при первом показе
        if self.stat_backend._value_text == "-":
            self._refresh()