from __future__ import annotations

import json

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.api_client import ApiClient
from app.core.worker import ApiWorker
from app.ui.activity import record_event
from app.ui.dialogs import show_error, show_warning
from app.ui.widgets import ActionButton, Card, ConsoleView, DimLabel, SectionHeading, StatusPill


class PersonsTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.api = ApiClient()
        self._worker: ApiWorker | None = None
        self._all_persons: list[dict] = []
        self._visible_persons: list[dict] = []
        self._selected_id: str | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(18)

        header_row = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.addWidget(SectionHeading("Database"))
        title_col.addWidget(DimLabel("Browse active records and inspect person metadata."))
        header_row.addLayout(title_col)
        header_row.addStretch()
        self.count_pill = StatusPill("0 records", state="idle")
        header_row.addWidget(self.count_pill)
        root.addLayout(header_row)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(10)
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter by label or person ID")
        self.filter_input.textChanged.connect(self._render_table)
        self.person_id_input = QLineEdit()
        self.person_id_input.setPlaceholderText("Open by person ID")
        self.get_btn = ActionButton("Open")
        self.get_btn.clicked.connect(self._load_selected_person)
        self.refresh_btn = ActionButton("Refresh", primary=True)
        self.refresh_btn.clicked.connect(self._load_list)
        filter_row.addWidget(self.filter_input, 2)
        filter_row.addWidget(self.person_id_input, 2)
        filter_row.addWidget(self.get_btn)
        filter_row.addWidget(self.refresh_btn)
        root.addLayout(filter_row)

        body_row = QHBoxLayout()
        body_row.setSpacing(18)

        list_card = Card()
        list_body = list_card.body()
        list_body.addWidget(SectionHeading("Records"))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Label", "Person ID", "Created"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        list_body.addWidget(self.table, 1)
        self.table_hint = DimLabel("Load records to inspect the database.")
        list_body.addWidget(self.table_hint)
        body_row.addWidget(list_card, 3)

        detail_card = Card()
        detail = detail_card.body()
        detail.addWidget(SectionHeading("Details"))
        self.selection_label = DimLabel("Select a row to inspect details.")
        detail.addWidget(self.selection_label)
        self.detail_console = ConsoleView("Selected person metadata will appear here.")
        self.detail_console.setMinimumHeight(380)
        detail.addWidget(self.detail_console)

        action_row = QHBoxLayout()
        self.delete_btn = ActionButton("Delete")
        self.delete_btn.clicked.connect(self._delete_selected)
        self.export_btn = ActionButton("Export JSON")
        self.export_btn.clicked.connect(self._export_detail)
        action_row.addWidget(self.delete_btn)
        action_row.addWidget(self.export_btn)
        action_row.addStretch()
        detail.addLayout(action_row)

        self.status_label = DimLabel("")
        detail.addWidget(self.status_label)
        detail.addStretch()
        body_row.addWidget(detail_card, 2)

        root.addLayout(body_row, 1)
        self._sync_action_state()

    def _run(self, func, *args, on_success) -> None:
        self.refresh_btn.setEnabled(False)
        self.get_btn.setEnabled(False)
        self.delete_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self._worker = ApiWorker(func, *args, parent=self)
        self._worker.finished.connect(on_success)
        self._worker.failed.connect(self._on_error)
        self._worker.start()

    def _sync_action_state(self) -> None:
        has_selection = bool(self._selected_id)
        self.delete_btn.setEnabled(has_selection)
        self.export_btn.setEnabled(has_selection and bool(self.detail_console.toPlainText().strip()))

    def _on_error(self, error: str) -> None:
        self.refresh_btn.setEnabled(True)
        self.get_btn.setEnabled(True)
        self._sync_action_state()
        self.status_label.setText("")
        record_event("database", "Registry request failed", severity="ERROR", details=error)
        show_error(self, "Database action failed", error)

    def _load_list(self) -> None:
        self.status_label.setText("Refreshing records...")
        self._run(self.api.list_persons, on_success=self._on_list_loaded)

    def _on_list_loaded(self, result: object) -> None:
        self.refresh_btn.setEnabled(True)
        self.get_btn.setEnabled(True)
        self.status_label.setText("")
        if not isinstance(result, list):
            return
        self._all_persons = result
        self.count_pill.set_state("ok", f"{len(result)} records")
        self._render_table()
        self._sync_action_state()
        record_event("database", f"Loaded {len(result)} profiles", severity="INFO")

    def _render_table(self) -> None:
        query = self.filter_input.text().strip().lower()
        self._visible_persons = []
        for person in self._all_persons:
            label = (person.get("label") or "").lower()
            person_id = str(person.get("id", "")).lower()
            if not query or query in label or query in person_id:
                self._visible_persons.append(person)

        self.table.setRowCount(len(self._visible_persons))
        for row, person in enumerate(self._visible_persons):
            created = str(person.get("created_at", ""))[:19].replace("T", " ")
            values = [
                person.get("label") or "Unnamed",
                str(person.get("id", ""))[:18],
                created,
            ]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(value))

        self.table_hint.setText(
            "No records match the current filter." if not self._visible_persons else f"Showing {len(self._visible_persons)} record(s)."
        )
        self.table.clearSelection()

    def _person_id(self) -> str | None:
        value = self.person_id_input.text().strip()
        if not value:
            show_warning(self, "Missing person ID", "Select a row or enter a person ID.")
            return None
        return value

    def _on_selection_changed(self) -> None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            self._selected_id = None
            self.selection_label.setText("Select a row to inspect details.")
            self.detail_console.setPlainText("")
            self._sync_action_state()
            return
        row = selected[0].row()
        if row >= len(self._visible_persons):
            return
        person_id = str(self._visible_persons[row].get("id", ""))
        self.person_id_input.setText(person_id)
        self._selected_id = person_id
        self._load_selected_person()

    def _load_selected_person(self) -> None:
        person_id = self._person_id()
        if not person_id:
            return
        self._selected_id = person_id
        self.status_label.setText("Loading profile...")
        self._run(self.api.get_person, person_id, on_success=self._on_person_loaded)

    def _on_person_loaded(self, result: object) -> None:
        self.refresh_btn.setEnabled(True)
        self.get_btn.setEnabled(True)
        self.status_label.setText("")
        if not isinstance(result, dict):
            return
        self.selection_label.setText(
            f"{result.get('label') or 'Unnamed'} · {str(result.get('id', ''))[:16]}"
        )
        self.detail_console.setPlainText(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        self._selected_id = str(result.get("id", "")) or self._selected_id
        self._sync_action_state()
        record_event(
            "database",
            f"Loaded profile {result.get('label') or result.get('id', '-')}",
            severity="INFO",
        )

    def _delete_selected(self) -> None:
        person_id = self._person_id()
        if not person_id:
            return
        confirm = QMessageBox.question(
            self,
            "Delete profile",
            f"Soft-delete profile {person_id[:16]}?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        self.status_label.setText("Deleting profile...")
        self._run(self.api.delete_person, person_id, on_success=self._on_deleted)

    def _on_deleted(self, result: object) -> None:
        self.refresh_btn.setEnabled(True)
        self.get_btn.setEnabled(True)
        self.status_label.setText("")
        self._selected_id = None
        self.selection_label.setText("Select a row to inspect details.")
        self.detail_console.setPlainText(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        self._sync_action_state()
        record_event("database", "Profile soft-deleted", severity="WARN", details=str(result))
        self._load_list()

    def _export_detail(self) -> None:
        content = self.detail_console.toPlainText().strip()
        if not content:
            show_warning(self, "Nothing to export", "Load a profile first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export person detail",
            "profile_detail.json",
            "JSON (*.json);;Text (*.txt)",
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        record_event("database", "Exported profile detail", severity="INFO", details=path)
        self.status_label.setText(f"Exported to {path}")
        self._sync_action_state()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._all_persons:
            self._load_list()

    def apply_global_filter(self, text: str) -> None:
        self.filter_input.setText(text)
