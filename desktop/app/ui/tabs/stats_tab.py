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


class StatsTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.api = ApiClient()
        self.stats_view = QTextEdit()
        self.stats_view.setReadOnly(True)

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
        layout.addWidget(self.stats_view)
        layout.addWidget(QLabel("Rebuild index (HNSW)"))
        layout.addLayout(form)
        layout.addWidget(rebuild_btn, alignment=Qt.AlignLeft)
        self.setLayout(layout)

    def _refresh(self) -> None:
        try:
            stats = self.api.index_stats()
            self.stats_view.setPlainText(str(stats))
        except Exception as exc:
            QMessageBox.critical(self, "Stats failed", str(exc))

    def _rebuild(self) -> None:
        try:
            params = {
                "m": int(self.m_input.text()),
                "ef_construction": int(self.efc_input.text()),
                "ef_search": int(self.efs_input.text()),
            }
            stats = self.api.rebuild_index("hnsw", params)
            self.stats_view.setPlainText(str(stats))
        except Exception as exc:
            QMessageBox.critical(self, "Rebuild failed", str(exc))
