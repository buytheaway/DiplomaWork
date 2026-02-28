"""Enrol tab — register a face image with an optional label."""

from __future__ import annotations

import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
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


class EnrollTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.api = ApiClient()
        self._worker: ApiWorker | None = None
        self.image_path = QLineEdit()
        self.label_input = QLineEdit()
        self.response_view = QTextEdit()
        self.response_view.setReadOnly(True)
        self.status_label = QLabel("")

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse)

        self.enroll_btn = QPushButton("Enroll")
        self.enroll_btn.clicked.connect(self._enroll)

        file_row = QHBoxLayout()
        file_row.addWidget(self.image_path)
        file_row.addWidget(browse_btn)

        form = QFormLayout()
        form.addRow(QLabel("Image"), file_row)
        form.addRow("Label", self.label_input)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self.enroll_btn, alignment=Qt.AlignLeft)
        layout.addWidget(self.status_label, alignment=Qt.AlignLeft)
        layout.addWidget(QLabel("Response"))
        layout.addWidget(self.response_view)
        self.setLayout(layout)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select image")
        if path:
            self.image_path.setText(path)

    def _enroll(self) -> None:
        path = self.image_path.text().strip()
        if not path:
            QMessageBox.warning(self, "Missing", "Select an image file")
            return
        label = self.label_input.text().strip() or None
        self.enroll_btn.setEnabled(False)
        self.status_label.setText("Enrolling…")
        self._worker = ApiWorker(self.api.enroll, path, label, parent=self)
        self._worker.finished.connect(self._on_success)
        self._worker.failed.connect(self._on_error)
        self._worker.start()

    def _on_success(self, result: object) -> None:
        self.enroll_btn.setEnabled(True)
        self.status_label.setText("")
        self.response_view.setPlainText(json.dumps(result, indent=2, default=str))

    def _on_error(self, error: str) -> None:
        self.enroll_btn.setEnabled(True)
        self.status_label.setText("")
        QMessageBox.critical(self, "Enroll failed", error)
