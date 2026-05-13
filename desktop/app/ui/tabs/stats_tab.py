from __future__ import annotations

import json
import time

from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.api_client import ApiClient
from app.core.worker import ApiWorker
from app.ui.activity import export_events_csv, recent_events, record_event
from app.ui.dialogs import show_error, show_warning
from app.ui.widgets import (
    ActionButton,
    Card,
    CollapsibleSection,
    ConsoleView,
    DimLabel,
    MetricCard,
    SectionHeading,
)


class StatsTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.api = ApiClient()
        self._worker: ApiWorker | None = None
        self._started = 0.0
        self._stats_payload: dict = {}
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(18)

        header = QVBoxLayout()
        header.addWidget(SectionHeading("Logs"))
        header.addWidget(DimLabel("Operational events, backend summary, and index maintenance tools."))
        root.addLayout(header)

        metrics = QHBoxLayout()
        metrics.setSpacing(14)
        self.metric_backend = MetricCard("Backend", "-")
        self.metric_default = MetricCard("Default pipeline", "-")
        self.metric_vectors = MetricCard("Indexed vectors", "-")
        metrics.addWidget(self.metric_backend, 1)
        metrics.addWidget(self.metric_default, 1)
        metrics.addWidget(self.metric_vectors, 1)
        root.addLayout(metrics)

        main_row = QHBoxLayout()
        main_row.setSpacing(18)

        left_card = Card()
        left = left_card.body()
        left_header = QHBoxLayout()
        left_header.addWidget(SectionHeading("Activity log"))
        left_header.addStretch()
        self.export_btn = ActionButton("Export CSV")
        self.export_btn.clicked.connect(self._export_events)
        left_header.addWidget(self.export_btn)
        left.addLayout(left_header)

        self.events_table = QTableWidget(0, 4)
        self.events_table.setHorizontalHeaderLabels(["Timestamp", "Severity", "Category", "Message"])
        self.events_table.verticalHeader().setVisible(False)
        self.events_table.setAlternatingRowColors(True)
        self.events_table.setSelectionBehavior(QTableWidget.SelectRows)
        left.addWidget(self.events_table, 1)
        main_row.addWidget(left_card, 3)

        right_col = QVBoxLayout()
        right_col.setSpacing(16)

        controls_card = CollapsibleSection("Index maintenance", expanded=False)
        controls = controls_card.body()
        controls.addWidget(DimLabel("Rebuild controls are kept secondary because logs are the main view."))

        pipeline_row = QHBoxLayout()
        self.pipeline_combo = QComboBox()
        self.pipeline_combo.addItem("Pretrained", "pretrained")
        self.pipeline_combo.addItem("Custom", "custom")
        self.refresh_btn = ActionButton("Refresh", primary=True)
        self.refresh_btn.clicked.connect(self._refresh)
        pipeline_row.addWidget(self.pipeline_combo)
        pipeline_row.addWidget(self.refresh_btn)
        controls.addLayout(pipeline_row)

        params_row = QHBoxLayout()
        params_row.setSpacing(10)
        self.m_input = QLineEdit("32")
        self.efc_input = QLineEdit("200")
        self.efs_input = QLineEdit("64")
        self.m_input.setPlaceholderText("HNSW M")
        self.efc_input.setPlaceholderText("efConstruction")
        self.efs_input.setPlaceholderText("efSearch")
        params_row.addWidget(self.m_input)
        params_row.addWidget(self.efc_input)
        params_row.addWidget(self.efs_input)
        controls.addLayout(params_row)

        self.rebuild_btn = ActionButton("Rebuild HNSW")
        self.rebuild_btn.clicked.connect(self._rebuild)
        controls.addWidget(self.rebuild_btn)

        self.latency_label = DimLabel("")
        controls.addWidget(self.latency_label)
        right_col.addWidget(controls_card)

        details_card = CollapsibleSection("Technical details", expanded=False)
        details = details_card.body()
        details.addWidget(DimLabel("Raw snapshot and rebuild payloads stay here as a secondary technical view."))
        self.stats_view = ConsoleView("Backend health and index payloads will appear here.")
        self.stats_view.setMinimumHeight(240)
        details.addWidget(self.stats_view)
        right_col.addWidget(details_card)

        main_row.addLayout(right_col, 2)
        root.addLayout(main_row, 1)

    def _snapshot(self) -> dict:
        health = self.api.health()
        stats = {}
        for pipeline in health.get("available_pipelines", []):
            stats[pipeline] = self.api.index_stats(pipeline)
        return {"health": health, "stats": stats}

    def _run(self, func, *args, on_success) -> None:
        self._started = time.perf_counter()
        self.refresh_btn.setEnabled(False)
        self.rebuild_btn.setEnabled(False)
        self._worker = ApiWorker(func, *args, parent=self)
        self._worker.finished.connect(on_success)
        self._worker.failed.connect(self._on_error)
        self._worker.start()

    def _refresh(self) -> None:
        self._run(self._snapshot, on_success=self._on_snapshot)

    def _on_snapshot(self, payload: object) -> None:
        self.refresh_btn.setEnabled(True)
        self.rebuild_btn.setEnabled(True)
        latency_ms = (time.perf_counter() - self._started) * 1000
        self.latency_label.setText(f"Last refresh: {latency_ms:.0f} ms")

        if not isinstance(payload, dict):
            return
        self._stats_payload = payload
        health = payload.get("health", {})
        stats = payload.get("stats", {})

        available = [str(v) for v in health.get("available_pipelines", [])]
        total_vectors = sum(int(stats.get(name, {}).get("embeddings_count", 0)) for name in available)
        self.metric_backend.set_value(health.get("embedding_backend", "-"), health.get("model_name", "-"))
        self.metric_default.set_value(str(health.get("default_pipeline", "-")).title(), ", ".join(available))
        self.metric_vectors.set_value(str(total_vectors), "Across loaded pipelines")

        self._render_events()
        self.stats_view.setPlainText(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        record_event("logs", "System snapshot refreshed", severity="INFO")

    def _render_events(self) -> None:
        events = recent_events(limit=50)
        self.events_table.setRowCount(len(events))
        for row, event in enumerate(events):
            self.events_table.setItem(row, 0, QTableWidgetItem(event.timestamp.strftime("%Y-%m-%d %H:%M:%S")))
            self.events_table.setItem(row, 1, QTableWidgetItem(event.severity))
            self.events_table.setItem(row, 2, QTableWidgetItem(event.category))
            self.events_table.setItem(row, 3, QTableWidgetItem(event.message))
        self.events_table.resizeColumnsToContents()

    def _rebuild(self) -> None:
        try:
            params = {
                "m": int(self.m_input.text()),
                "ef_construction": int(self.efc_input.text()),
                "ef_search": int(self.efs_input.text()),
            }
        except ValueError:
            show_warning(self, "Invalid params", "HNSW parameters must be integers.")
            return
        pipeline = self.pipeline_combo.currentData()
        self._run(self.api.rebuild_index, "hnsw", params, pipeline, on_success=self._on_rebuild)

    def _on_rebuild(self, payload: object) -> None:
        self.refresh_btn.setEnabled(True)
        self.rebuild_btn.setEnabled(True)
        latency_ms = (time.perf_counter() - self._started) * 1000
        self.latency_label.setText(f"Rebuild completed in {latency_ms:.0f} ms")
        self.stats_view.setPlainText(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        record_event(
            "logs",
            f"Rebuilt {self.pipeline_combo.currentData()} index",
            severity="WARN",
            details=json.dumps(payload, ensure_ascii=False),
        )
        self._refresh()

    def _on_error(self, error: str) -> None:
        self.refresh_btn.setEnabled(True)
        self.rebuild_btn.setEnabled(True)
        record_event("logs", "System request failed", severity="ERROR", details=error)
        show_error(self, "System request failed", error)

    def _export_events(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export event log",
            "activity_log.csv",
            "CSV (*.csv)",
        )
        if not path:
            return
        out_path = export_events_csv(path)
        self.latency_label.setText(f"Exported log to {out_path}")
        record_event("logs", "Exported activity log", severity="INFO", details=str(out_path))
        self._render_events()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._render_events()
        if not self._stats_payload:
            self._refresh()

    def apply_global_filter(self, text: str) -> None:
        query = text.strip().lower()
        events = recent_events(limit=50)
        filtered = [
            event
            for event in events
            if not query
            or query in event.category.lower()
            or query in event.message.lower()
            or query in event.severity.lower()
        ]
        self.events_table.setRowCount(len(filtered))
        for row, event in enumerate(filtered):
            self.events_table.setItem(row, 0, QTableWidgetItem(event.timestamp.strftime("%Y-%m-%d %H:%M:%S")))
            self.events_table.setItem(row, 1, QTableWidgetItem(event.severity))
            self.events_table.setItem(row, 2, QTableWidgetItem(event.category))
            self.events_table.setItem(row, 3, QTableWidgetItem(event.message))
        self.events_table.resizeColumnsToContents()
