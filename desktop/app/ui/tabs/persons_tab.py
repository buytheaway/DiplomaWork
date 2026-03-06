from __future__ import annotations

import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.api_client import ApiClient
from app.core.worker import ApiWorker
from app.ui.widgets import Card, DimLabel, SectionHeading


class PersonsTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.api = ApiClient()
        self._worker: ApiWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        # --- список персон ---
        list_card = Card()
        lc = list_card.body()

        header_row = QHBoxLayout()
        header_row.addWidget(SectionHeading("Enrolled persons"))
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setFixedWidth(90)
        self.refresh_btn.clicked.connect(self._load_list)
        header_row.addStretch()
        header_row.addWidget(self.refresh_btn)
        lc.addLayout(header_row)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Person ID", "Label", "Created"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.setMinimumHeight(160)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        lc.addWidget(self.table)

        self.count_label = DimLabel("0 persons")
        lc.addWidget(self.count_label)

        root.addWidget(list_card, 2)

        # --- details + actions ---
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(16)

        # Detail card
        detail_card = Card()
        dc = detail_card.body()
        dc.addWidget(SectionHeading("Person detail"))

        id_row = QHBoxLayout()
        id_row.setSpacing(8)
        id_row.addWidget(DimLabel("Person ID"))
        self.person_id_input = QLineEdit()
        self.person_id_input.setPlaceholderText("Select from list or enter UUID")
        id_row.addWidget(self.person_id_input, 1)

        self.get_btn = QPushButton("Get")
        self.get_btn.setFixedWidth(70)
        self.get_btn.clicked.connect(self._get_person)
        id_row.addWidget(self.get_btn)

        dc.addLayout(id_row)

        self.response_view = QTextEdit()
        self.response_view.setReadOnly(True)
        self.response_view.setMaximumHeight(140)
        self.response_view.setPlaceholderText("Person detail will appear here...")
        dc.addWidget(self.response_view)

        bottom_row.addWidget(detail_card, 3)

        # Action card
        action_card = Card()
        ac = action_card.body()
        ac.addWidget(SectionHeading("Actions"))
        ac.addSpacing(8)

        self.delete_btn = QPushButton("Delete person")
        self.delete_btn.setObjectName("danger")
        self.delete_btn.clicked.connect(self._delete_person)
        ac.addWidget(self.delete_btn)

        ac.addSpacing(4)
        ac.addWidget(DimLabel("Soft-deletes the person and\ndeactivates all embeddings."))
        ac.addStretch()

        self.status_label = DimLabel("")
        ac.addWidget(self.status_label)

        bottom_row.addWidget(action_card, 1)

        root.addLayout(bottom_row, 1)

    # --- list ---

    def _load_list(self) -> None:
        self.refresh_btn.setEnabled(False)
        self._worker = ApiWorker(self.api.list_persons, parent=self)
        self._worker.finished.connect(self._on_list_loaded)
        self._worker.failed.connect(self._on_error)
        self._worker.start()

    def _on_list_loaded(self, result: object) -> None:
        self.refresh_btn.setEnabled(True)
        if not isinstance(result, list):
            return
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(result))
        for row, person in enumerate(result):
            self.table.setItem(row, 0, QTableWidgetItem(person.get("id", "")))
            self.table.setItem(row, 1, QTableWidgetItem(person.get("label") or ""))
            created = person.get("created_at", "")
            if isinstance(created, str) and len(created) > 19:
                created = created[:19].replace("T", " ")
            self.table.setItem(row, 2, QTableWidgetItem(str(created)))
        self.table.setSortingEnabled(True)
        self.count_label.setText(f"{len(result)} persons")

    def _on_row_selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            pid_item = self.table.item(rows[0].row(), 0)
            if pid_item:
                self.person_id_input.setText(pid_item.text())

    # --- detail / delete ---

    def _person_id(self) -> str | None:
        pid = self.person_id_input.text().strip()
        if not pid:
            QMessageBox.warning(self, "Missing", "Enter a Person ID")
            return None
        return pid

    def _run(self, func, *args) -> None:
        self.status_label.setText("Loading...")
        self._worker = ApiWorker(func, *args, parent=self)
        self._worker.finished.connect(self._on_detail_success)
        self._worker.failed.connect(self._on_error)
        self._worker.start()

    def _on_detail_success(self, result: object) -> None:
        self.status_label.setText("")
        self.response_view.setPlainText(json.dumps(result, indent=2, default=str))

    def _on_error(self, error: str) -> None:
        self.refresh_btn.setEnabled(True)
        self.status_label.setText("")
        QMessageBox.critical(self, "Error", error)

    def _get_person(self) -> None:
        pid = self._person_id()
        if pid:
            self._run(self.api.get_person, pid)

    def _delete_person(self) -> None:
        pid = self._person_id()
        if not pid:
            return
        confirm = QMessageBox.question(
            self, "Confirm delete",
            f"Delete person {pid[:12]}...?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm == QMessageBox.Yes:
            self._run(self.api.delete_person, pid)

    def showEvent(self, event) -> None:
        # Автозагрузка при переключении на вкладку
        super().showEvent(event)
        if self.table.rowCount() == 0:
            self._load_list()