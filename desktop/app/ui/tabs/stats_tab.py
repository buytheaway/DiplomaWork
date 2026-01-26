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

from app.core.api_client import ApiClient, format_api_error


class StatsTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.api = ApiClient()
        self.stats_view = QTextEdit()
        self.stats_view.setReadOnly(True)
        self.last_rebuild_params: dict[str, int] | None = None
        self.last_rebuild_label = QLabel("Last rebuild params: -")
        self.latency_label = QLabel("Latency: - ms")

        self.m_input = QLineEdit("32")
        self.efc_input = QLineEdit("200")
        self.efs_input = QLineEdit("64")

        refresh_btn = QPushButton("Refresh stats")
        refresh_btn.clicked.connect(self._refresh)

        rebuild_btn = QPushButton("Rebuild HNSW")
        rebuild_btn.clicked.connect(self._rebuild)

        form = QFormLayout()
        form.addRow(QLabel("HNSW M"), self.m_input)
        form.addRow(QLabel("HNSW efConstruction"), self.efc_input)
        form.addRow(QLabel("HNSW efSearch"), self.efs_input)

        layout = QVBoxLayout()
        layout.addWidget(refresh_btn, alignment=Qt.AlignLeft)
        layout.addWidget(self.latency_label, alignment=Qt.AlignLeft)
        layout.addWidget(self.stats_view)
        layout.addWidget(self.last_rebuild_label)
        layout.addWidget(QLabel("Rebuild index (HNSW)"))
        layout.addLayout(form)
        layout.addWidget(rebuild_btn, alignment=Qt.AlignLeft)
        self.setLayout(layout)

    def _refresh(self) -> None:
        try:
            start = time.perf_counter()
            stats = self.api.index_stats()
            latency_ms = (time.perf_counter() - start) * 1000
            self.latency_label.setText(f"Latency: {latency_ms:.2f} ms")
            self.stats_view.setPlainText(json.dumps(stats, indent=2))
        except Exception as exc:
            QMessageBox.critical(self, "Stats failed", format_api_error(exc))

    def _rebuild(self) -> None:
        try:
            params = {
                "m": int(self.m_input.text()),
                "ef_construction": int(self.efc_input.text()),
                "ef_search": int(self.efs_input.text()),
            }
            start = time.perf_counter()
            stats = self.api.rebuild_index("hnsw", params)
            latency_ms = (time.perf_counter() - start) * 1000
            self.latency_label.setText(f"Latency: {latency_ms:.2f} ms")
            self.last_rebuild_params = params
            self.last_rebuild_label.setText(
                f"Last rebuild params: {json.dumps(self.last_rebuild_params)}"
            )
            self.stats_view.setPlainText(json.dumps(stats, indent=2))
        except Exception as exc:
            QMessageBox.critical(self, "Rebuild failed", format_api_error(exc))
