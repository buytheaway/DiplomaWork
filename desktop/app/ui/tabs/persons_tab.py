from __future__ import annotations

import json

from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
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
from app.ui.widgets import (
    ActionButton,
    Card,
    CollapsibleSection,
    ConsoleView,
    DimLabel,
    InfoRow,
    SectionHeading,
    StatusPill,
)


class PersonsTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.api = ApiClient()
        self._worker: ApiWorker | None = None
        self._current_items: list[dict] = []
        self._visible_persons: list[dict] = []
        self._current_offset = 0
        self._page_size = 200
        self._total_count = 0
        self._templates_count = 0
        self._indexed_count = 0
        self._search_query = ""
        self._loaded_once = False
        self._selected_id: str | None = None
        self._detail_payload = ""
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
        self.templates_pill = StatusPill("0 templates", state="idle")
        self.indexed_pill = StatusPill("0 indexed", state="idle")
        header_row.addWidget(self.count_pill)
        header_row.addWidget(self.templates_pill)
        header_row.addWidget(self.indexed_pill)
        root.addLayout(header_row)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(10)
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Search all records by label or ID")
        self.filter_input.returnPressed.connect(self._apply_search)
        self.search_btn = ActionButton("Search")
        self.search_btn.clicked.connect(self._apply_search)
        self.clear_search_btn = ActionButton("Clear search")
        self.clear_search_btn.clicked.connect(self._clear_search)
        self.person_id_input = QLineEdit()
        self.person_id_input.setPlaceholderText("Open by person ID")
        self.get_btn = ActionButton("Open")
        self.get_btn.clicked.connect(self._load_selected_person)
        self.refresh_btn = ActionButton("Refresh", primary=True)
        self.refresh_btn.clicked.connect(self._load_list)
        filter_row.addWidget(self.filter_input, 2)
        filter_row.addWidget(self.search_btn)
        filter_row.addWidget(self.clear_search_btn)
        filter_row.addWidget(self.person_id_input, 2)
        filter_row.addWidget(self.get_btn)
        filter_row.addWidget(self.refresh_btn)
        root.addLayout(filter_row)

        page_row = QHBoxLayout()
        page_row.setSpacing(10)
        self.prev_btn = ActionButton("Previous")
        self.prev_btn.clicked.connect(self._load_previous_page)
        self.next_btn = ActionButton("Next")
        self.next_btn.clicked.connect(self._load_next_page)
        self.page_size_combo = QComboBox()
        self.page_size_combo.addItems(["50", "100", "200", "500"])
        self.page_size_combo.setCurrentText(str(self._page_size))
        self.page_size_combo.currentTextChanged.connect(self._on_page_size_changed)
        self.page_input = QLineEdit()
        self.page_input.setPlaceholderText("Page")
        self.page_input.returnPressed.connect(self._jump_to_page)
        self.go_page_btn = ActionButton("Go")
        self.go_page_btn.clicked.connect(self._jump_to_page)
        self.page_status = DimLabel("No records loaded.")
        page_row.addWidget(self.prev_btn)
        page_row.addWidget(self.next_btn)
        page_row.addWidget(DimLabel("Page size"))
        page_row.addWidget(self.page_size_combo)
        page_row.addWidget(DimLabel("Jump"))
        page_row.addWidget(self.page_input)
        page_row.addWidget(self.go_page_btn)
        page_row.addWidget(self.page_status, 1)
        root.addLayout(page_row)

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

        detail_grid = QGridLayout()
        detail_grid.setHorizontalSpacing(18)
        detail_grid.setVerticalSpacing(12)
        self.detail_label_info = InfoRow("Label", "-")
        self.detail_status_info = InfoRow("Status", "-")
        self.detail_id_info = InfoRow("Person ID", "-")
        self.detail_created_info = InfoRow("Created", "-")
        self.detail_embeddings_info = InfoRow("Embeddings", "-")
        self.detail_models_info = InfoRow("Models", "-")
        detail_grid.addWidget(self.detail_label_info, 0, 0)
        detail_grid.addWidget(self.detail_status_info, 0, 1)
        detail_grid.addWidget(self.detail_id_info, 1, 0, 1, 2)
        detail_grid.addWidget(self.detail_created_info, 2, 0)
        detail_grid.addWidget(self.detail_embeddings_info, 2, 1)
        detail_grid.addWidget(self.detail_models_info, 3, 0, 1, 2)
        detail.addLayout(detail_grid)

        raw_details = CollapsibleSection("Technical details", expanded=False)
        raw_body = raw_details.body()
        raw_body.addWidget(DimLabel("Raw person payload is available for export and debugging."))
        self.detail_console = ConsoleView("Selected person metadata will appear here.")
        self.detail_console.setMinimumHeight(220)
        raw_body.addWidget(self.detail_console)
        detail.addWidget(raw_details)

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
        self.search_btn.setEnabled(False)
        self.clear_search_btn.setEnabled(False)
        self.get_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.page_size_combo.setEnabled(False)
        self.page_input.setEnabled(False)
        self.go_page_btn.setEnabled(False)
        self.delete_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self._worker = ApiWorker(func, *args, parent=self)
        self._worker.finished.connect(on_success)
        self._worker.failed.connect(self._on_error)
        self._worker.start()

    def _sync_action_state(self) -> None:
        has_selection = bool(self._selected_id)
        self.delete_btn.setEnabled(has_selection)
        self.export_btn.setEnabled(has_selection and bool(self._detail_payload.strip()))
        self._sync_paging_state()

    def _sync_paging_state(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self.refresh_btn.setEnabled(True)
        self.search_btn.setEnabled(True)
        self.clear_search_btn.setEnabled(True)
        self.get_btn.setEnabled(True)
        self.page_size_combo.setEnabled(True)
        self.page_input.setEnabled(True)
        self.go_page_btn.setEnabled(self._total_count > 0)
        self.prev_btn.setEnabled(self._current_offset > 0)
        self.next_btn.setEnabled(self._current_offset + self._page_size < self._total_count)

    def _current_page(self) -> int:
        return self._current_offset // max(1, self._page_size) + 1

    def _total_pages(self) -> int:
        if self._total_count <= 0:
            return 1
        return (self._total_count - 1) // max(1, self._page_size) + 1

    def _page_range_text(self) -> str:
        if self._total_count <= 0:
            return "No matching records" if self._search_query else "No records"
        first = self._current_offset + 1
        last = min(self._current_offset + len(self._current_items), self._total_count)
        noun = "matching records" if self._search_query else "records"
        return (
            f"Showing {first:,}-{last:,} of {self._total_count:,} {noun} "
            f"- Page {self._current_page():,} of {self._total_pages():,}"
        )

    def _clear_detail(self, message: str = "Select a row to inspect details.") -> None:
        self.selection_label.setText(message)
        self.detail_label_info.set_value("-")
        self.detail_status_info.set_value("-")
        self.detail_id_info.set_value("-")
        self.detail_created_info.set_value("-")
        self.detail_embeddings_info.set_value("-")
        self.detail_models_info.set_value("-")
        self.detail_console.setPlainText("")
        self._detail_payload = ""

    def _on_error(self, error: str) -> None:
        self.refresh_btn.setEnabled(True)
        self.search_btn.setEnabled(True)
        self.clear_search_btn.setEnabled(True)
        self.get_btn.setEnabled(True)
        self._sync_action_state()
        self.page_status.setText(f"Failed to load records: {error}")
        self.status_label.setText("")
        record_event("database", "Registry request failed", severity="ERROR", details=error)
        show_error(self, "Database action failed", error)

    def _load_list(self) -> None:
        self.status_label.setText("Refreshing records...")
        self._run(
            self._load_database_page,
            on_success=self._on_list_loaded,
        )

    def _load_database_page(self) -> dict:
        persons = self.api.list_persons(
            self._page_size,
            self._current_offset,
            self._search_query or None,
        )
        database_stats = self.api.database_stats()
        health = self.api.health()
        indexed_count = 0
        for pipeline in health.get("available_pipelines", []):
            stats = self.api.index_stats(str(pipeline))
            indexed_count += int(stats.get("embeddings_count", 0) or 0)
        return {
            "persons": persons,
            "database_stats": database_stats,
            "indexed_count": indexed_count,
        }

    def _load_previous_page(self) -> None:
        self._current_offset = max(0, self._current_offset - self._page_size)
        self._load_list()

    def _load_next_page(self) -> None:
        if self._current_offset + self._page_size >= self._total_count:
            return
        self._current_offset += self._page_size
        self._load_list()

    def _on_page_size_changed(self, value: str) -> None:
        try:
            self._page_size = int(value)
        except ValueError:
            self._page_size = 200
        self._current_offset = 0
        if self._loaded_once:
            self._load_list()

    def _apply_search(self) -> None:
        self._search_query = self.filter_input.text().strip()
        self._current_offset = 0
        self._load_list()

    def _clear_search(self) -> None:
        if not self.filter_input.text().strip() and not self._search_query:
            return
        self.filter_input.setText("")
        self._search_query = ""
        self._current_offset = 0
        self._load_list()

    def _jump_to_page(self) -> None:
        raw_value = self.page_input.text().strip()
        if not raw_value:
            return
        try:
            requested_page = int(raw_value)
        except ValueError:
            requested_page = self._current_page()
        target_page = max(1, min(requested_page, self._total_pages()))
        self.page_input.setText(str(target_page))
        self._current_offset = (target_page - 1) * self._page_size
        self._load_list()

    def _on_list_loaded(self, result: object) -> None:
        self.refresh_btn.setEnabled(True)
        self.search_btn.setEnabled(True)
        self.clear_search_btn.setEnabled(True)
        self.get_btn.setEnabled(True)
        self.page_size_combo.setEnabled(True)
        self.page_input.setEnabled(True)
        self.status_label.setText("")
        if not isinstance(result, dict):
            return
        persons_payload = result.get("persons", result)
        if not isinstance(persons_payload, dict):
            return
        database_stats = result.get("database_stats", {})
        if not isinstance(database_stats, dict):
            database_stats = {}
        self._templates_count = int(database_stats.get("active_embeddings", 0) or 0)
        self._indexed_count = int(result.get("indexed_count", 0) or 0)

        items = persons_payload.get("items", [])
        self._current_items = items if isinstance(items, list) else []
        self._total_count = int(persons_payload.get("total", 0) or 0)
        self._page_size = int(persons_payload.get("limit", self._page_size) or self._page_size)
        self._current_offset = int(persons_payload.get("offset", self._current_offset) or 0)
        self._loaded_once = True
        if (
            self._total_count > 0
            and not self._current_items
            and self._current_offset >= self._total_count
        ):
            self._current_offset = ((self._total_count - 1) // self._page_size) * self._page_size
            self._load_list()
            return
        self.count_pill.set_state("ok", f"{self._total_count:,} identities")
        self.templates_pill.set_state("ok", f"{self._templates_count:,} templates")
        self.indexed_pill.set_state("ok", f"{self._indexed_count:,} indexed")
        self.page_input.setText(str(self._current_page()))
        self._render_table()
        self._sync_action_state()
        self.page_status.setText(self._page_range_text())
        record_event(
            "database",
            f"Loaded {len(self._current_items)} of {self._total_count} profiles",
            severity="INFO",
        )

    def _render_table(self) -> None:
        self._visible_persons = list(self._current_items)

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

        if not self._visible_persons:
            hint = "No records match the current search." if self._search_query else "No records"
        else:
            hint = self._page_range_text()
        self.table_hint.setText(hint)
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
            self._clear_detail()
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
        label = result.get("label") or "Unnamed"
        person_id = str(result.get("id", ""))
        status = str(result.get("status", "-"))
        created = str(result.get("created_at", ""))[:19].replace("T", " ")
        embeddings = result.get("embeddings", [])
        active_count = sum(1 for item in embeddings if item.get("is_active"))
        models = sorted({str(item.get("model", "-")) for item in embeddings})

        self.selection_label.setText(
            f"{label} - {person_id[:16]}"
        )
        self.detail_label_info.set_value(str(label))
        self.detail_status_info.set_value(status.title())
        self.detail_id_info.set_value(person_id or "-")
        self.detail_created_info.set_value(created or "-")
        self.detail_embeddings_info.set_value(f"{active_count}/{len(embeddings)} active")
        self.detail_models_info.set_value(", ".join(models) if models else "-")
        self._detail_payload = json.dumps(result, indent=2, ensure_ascii=False, default=str)
        self.detail_console.setPlainText(self._detail_payload)
        self._selected_id = person_id or self._selected_id
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
        self._clear_detail("Profile deleted. Select another row to inspect details.")
        self.detail_console.setPlainText(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        self._sync_action_state()
        record_event("database", "Profile soft-deleted", severity="WARN", details=str(result))
        self._load_list()

    def _export_detail(self) -> None:
        content = self._detail_payload.strip()
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
        if not self._loaded_once:
            self._load_list()

    def apply_global_filter(self, text: str) -> None:
        self.filter_input.setText(text)
        if self._loaded_once:
            self._apply_search()
