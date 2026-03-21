from __future__ import annotations

import json

from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.core.api_client import ApiClient
from app.core.worker import ApiWorker
from app.ui.activity import record_event
from app.ui.dialogs import show_error, show_warning
from app.ui.widgets import ActionButton, Card, ConsoleView, DimLabel, PersonCard, SectionHeading, StatusPill


class PersonsTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.api = ApiClient()
        self._worker: ApiWorker | None = None
        self._all_persons: list[dict] = []
        self._selected_id: str | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(18)

        header_row = QHBoxLayout()
        title_col = QVBoxLayout()
        title = SectionHeading("Database")
        subtitle = DimLabel("Person records and embedding metadata")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        header_row.addLayout(title_col)
        header_row.addStretch()
        self.count_pill = StatusPill("0 RECORDS", state="idle")
        header_row.addWidget(self.count_pill)
        root.addLayout(header_row)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(10)
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter by label or person id")
        self.filter_input.textChanged.connect(self._render_cards)
        self.person_id_input = QLineEdit()
        self.person_id_input.setPlaceholderText("Person id")
        self.get_btn = ActionButton("Load person")
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

        cards_card = Card()
        cards_body = cards_card.body()
        cards_body.addWidget(SectionHeading("People"))
        self.cards_scroll = QScrollArea()
        self.cards_scroll.setWidgetResizable(True)
        self.cards_scroll.setFrameShape(QScrollArea.NoFrame)
        self.cards_host = QWidget()
        self.cards_layout = QGridLayout(self.cards_host)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setHorizontalSpacing(14)
        self.cards_layout.setVerticalSpacing(14)
        self.cards_scroll.setWidget(self.cards_host)
        cards_body.addWidget(self.cards_scroll, 1)
        body_row.addWidget(cards_card, 3)

        detail_card = Card()
        detail = detail_card.body()
        detail.addWidget(SectionHeading("Details"))
        self.selection_label = DimLabel("Select a person card to inspect details.")
        detail.addWidget(self.selection_label)

        self.detail_console = ConsoleView("Selected person metadata will appear here.")
        self.detail_console.setMinimumHeight(360)
        detail.addWidget(self.detail_console)

        action_row = QHBoxLayout()
        self.delete_btn = ActionButton("Delete")
        self.delete_btn.clicked.connect(self._delete_selected)
        self.export_btn = ActionButton("Export JSON")
        self.export_btn.clicked.connect(self._export_detail)
        action_row.addWidget(self.delete_btn)
        action_row.addWidget(self.export_btn)
        detail.addLayout(action_row)

        self.status_label = DimLabel("")
        detail.addWidget(self.status_label)
        detail.addStretch()
        body_row.addWidget(detail_card, 2)

        root.addLayout(body_row, 1)

    def _run(self, func, *args, on_success) -> None:
        self.refresh_btn.setEnabled(False)
        self.get_btn.setEnabled(False)
        self.delete_btn.setEnabled(False)
        self._worker = ApiWorker(func, *args, parent=self)
        self._worker.finished.connect(on_success)
        self._worker.failed.connect(self._on_error)
        self._worker.start()

    def _on_error(self, error: str) -> None:
        self.refresh_btn.setEnabled(True)
        self.get_btn.setEnabled(True)
        self.delete_btn.setEnabled(True)
        self.status_label.setText("")
        record_event("database", "Registry request failed", severity="ERROR", details=error)
        show_error(self, "Database action failed", error)

    def _load_list(self) -> None:
        self.status_label.setText("Refreshing...")
        self._run(self.api.list_persons, on_success=self._on_list_loaded)

    def _on_list_loaded(self, result: object) -> None:
        self.refresh_btn.setEnabled(True)
        self.get_btn.setEnabled(True)
        self.delete_btn.setEnabled(True)
        self.status_label.setText("")
        if not isinstance(result, list):
            return
        self._all_persons = result
        self.count_pill.set_state("ok", f"{len(result)} RECORDS")
        self._render_cards()
        record_event("database", f"Loaded {len(result)} profiles", severity="INFO")

    def _render_cards(self) -> None:
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        query = self.filter_input.text().strip().lower()
        filtered = []
        for person in self._all_persons:
            label = (person.get("label") or "").lower()
            person_id = str(person.get("id", "")).lower()
            if not query or query in label or query in person_id:
                filtered.append(person)

        if not filtered:
            self.cards_layout.addWidget(DimLabel("No profiles match the current filter."), 0, 0)
            return

        for index, person in enumerate(filtered):
            row, col = divmod(index, 3)
            card = PersonCard(person)
            card.selected.connect(self._load_person_by_id)
            card.deleteRequested.connect(self._confirm_delete)
            self.cards_layout.addWidget(card, row, col)

    def _person_id(self) -> str | None:
        value = self.person_id_input.text().strip()
        if not value:
            show_warning(self, "Missing person id", "Select a card or type a person id.")
            return None
        return value

    def _load_person_by_id(self, person_id: str) -> None:
        self.person_id_input.setText(person_id)
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
        self.delete_btn.setEnabled(True)
        self.status_label.setText("")
        if not isinstance(result, dict):
            return
        self.selection_label.setText(
            f"Selected  {result.get('label') or 'UNNAMED'}  /  {str(result.get('id', ''))[:16]}"
        )
        self.detail_console.setPlainText(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        record_event(
            "database",
            f"Loaded profile {result.get('label') or result.get('id', '-')}",
            severity="INFO",
        )

    def _confirm_delete(self, person_id: str) -> None:
        self.person_id_input.setText(person_id)
        self._delete_selected()

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
        self.delete_btn.setEnabled(True)
        self.status_label.setText("")
        record_event("database", "Profile soft-deleted", severity="WARN", details=str(result))
        self._load_list()
        self.detail_console.setPlainText(json.dumps(result, indent=2, ensure_ascii=False, default=str))

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

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._all_persons:
            self._load_list()

    def apply_global_filter(self, text: str) -> None:
        self.filter_input.setText(text)
