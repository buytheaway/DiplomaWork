"""Persons management tab — view and delete enrolled persons."""

from __future__ import annotations

import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
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


class PersonsTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.api = ApiClient()
        self._worker: ApiWorker | None = None

        self.person_id_input = QLineEdit()
        self.person_id_input.setPlaceholderText("Enter Person ID (UUID)")
        self.response_view = QTextEdit()
        self.response_view.setReadOnly(True)
        self.status_label = QLabel("")

        get_btn = QPushButton("Get Person")
        get_btn.clicked.connect(self._get_person)

        delete_btn = QPushButton("Delete Person")
        delete_btn.clicked.connect(self._delete_person)

        btn_row = QHBoxLayout()
        btn_row.addWidget(get_btn)
        btn_row.addWidget(delete_btn)

        form = QFormLayout()
        form.addRow("Person ID", self.person_id_input)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addLayout(btn_row)
        layout.addWidget(self.status_label, alignment=Qt.AlignLeft)
        layout.addWidget(QLabel("Response"))
        layout.addWidget(self.response_view)
        self.setLayout(layout)

    # ── helpers ───────────────────────────────────────────────────────────

    def _person_id(self) -> str | None:
        pid = self.person_id_input.text().strip()
        if not pid:
            QMessageBox.warning(self, "Missing", "Enter a Person ID")
            return None
        return pid

    def _run(self, func, *args) -> None:
        self.status_label.setText("Loading…")
        self._worker = ApiWorker(func, *args, parent=self)
        self._worker.finished.connect(self._on_success)
        self._worker.failed.connect(self._on_error)
        self._worker.start()

    def _on_success(self, result: object) -> None:
        self.status_label.setText("")
        self.response_view.setPlainText(json.dumps(result, indent=2, default=str))

    def _on_error(self, error: str) -> None:
        self.status_label.setText("")
        QMessageBox.critical(self, "Error", error)

    # ── actions ──────────────────────────────────────────────────────────

    def _get_person(self) -> None:
        pid = self._person_id()
        if pid:
            self._run(self.api.get_person, pid)

    def _delete_person(self) -> None:
        pid = self._person_id()
        if pid:
            self._run(self.api.delete_person, pid)
