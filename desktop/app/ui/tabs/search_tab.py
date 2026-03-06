# Главная вкладка — поиск лица по индексу
from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.api_client import ApiClient
from app.core.worker import ApiWorker
from app.ui.widgets import Card, DimLabel, ImagePreview, InfoRow, SectionHeading


class NumericItem(QTableWidgetItem):
    def __init__(self, value: float | None) -> None:
        text = f"{value:.4f}" if value is not None else ""
        super().__init__(text)
        self.sort_value = value if value is not None else float("-inf")

    def __lt__(self, other: QTableWidgetItem) -> bool:
        if isinstance(other, NumericItem):
            return self.sort_value < other.sort_value
        return super().__lt__(other)


class SearchTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.api = ApiClient()
        self._worker: ApiWorker | None = None
        self._search_start: float = 0.0
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        # --- верхний ряд: форма + превью ---
        top_row = QHBoxLayout()
        top_row.setSpacing(16)

        # Форма поиска
        form_card = Card()
        fc = form_card.body()
        fc.addWidget(SectionHeading("Face query"))

        fc.addWidget(DimLabel("Image file"))
        file_row = QHBoxLayout()
        file_row.setSpacing(8)
        self.image_path = QLineEdit()
        self.image_path.setPlaceholderText("Select a face image...")
        self.image_path.setReadOnly(True)
        file_row.addWidget(self.image_path, 1)

        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse)
        file_row.addWidget(browse_btn)
        fc.addLayout(file_row)

        fc.addSpacing(6)

        k_row = QHBoxLayout()
        k_row.setSpacing(8)
        k_row.addWidget(DimLabel("Top K"))
        self.k_input = QSpinBox()
        self.k_input.setRange(1, 100)
        self.k_input.setValue(5)
        self.k_input.setFixedWidth(70)
        k_row.addWidget(self.k_input)
        k_row.addStretch()
        fc.addLayout(k_row)

        fc.addSpacing(10)

        self.search_btn = QPushButton("Search")
        self.search_btn.setObjectName("primary")
        self.search_btn.setFixedHeight(36)
        self.search_btn.clicked.connect(self._search)
        fc.addWidget(self.search_btn)
        fc.addStretch()

        top_row.addWidget(form_card, 3)

        # Превью + результат
        right_col = QVBoxLayout()
        right_col.setSpacing(12)

        preview_card = Card()
        pc = preview_card.body()
        pc.setAlignment(Qt.AlignCenter)
        self.preview = ImagePreview(160)
        pc.addWidget(self.preview, 0, Qt.AlignCenter)
        right_col.addWidget(preview_card)

        # Бейдж решения
        self.decision_badge = QLabel("—")
        self.decision_badge.setAlignment(Qt.AlignCenter)
        self.decision_badge.setFixedHeight(32)
        self.decision_badge.setStyleSheet(
            "background-color: #2c2d31; color: #8b8d93; border-radius: 4px;"
            " font-weight: 700; font-size: 15px;"
        )
        right_col.addWidget(self.decision_badge)

        right_col.addStretch()
        top_row.addLayout(right_col, 1)

        root.addLayout(top_row)

        # --- сводка ---
        summary_card = Card()
        sc = summary_card.body()
        info_row = QHBoxLayout()
        info_row.setSpacing(24)
        self.info_score = InfoRow("Best score")
        self.info_threshold = InfoRow("Threshold")
        self.info_latency = InfoRow("Latency")
        self.info_matches = InfoRow("Matches")
        for w in [self.info_score, self.info_threshold, self.info_latency, self.info_matches]:
            info_row.addWidget(w)
        info_row.addStretch()
        sc.addLayout(info_row)
        root.addWidget(summary_card)

        # --- таблица результатов ---
        results_card = Card()
        rc = results_card.body()
        rc.addWidget(SectionHeading("Search results"))

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Person ID", "Label", "Score", "Distance"])
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        rc.addWidget(self.table)

        root.addWidget(results_card, 1)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select image", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All files (*.*)",
        )
        if path:
            self.image_path.setText(path)
            self.preview.load(path)

    def _search(self) -> None:
        path = self.image_path.text().strip()
        if not path:
            QMessageBox.warning(self, "Missing", "Select an image file")
            return
        k = self.k_input.value()
        self.search_btn.setEnabled(False)
        self._search_start = time.perf_counter()
        self._worker = ApiWorker(self.api.search, path, k, parent=self)
        self._worker.finished.connect(self._on_success)
        self._worker.failed.connect(self._on_error)
        self._worker.start()

    def _on_success(self, response: object) -> None:
        self.search_btn.setEnabled(True)
        latency_ms = (time.perf_counter() - self._search_start) * 1000

        if isinstance(response, dict):
            results = response.get("results", [])
            decision = response.get("decision", "unknown")
            threshold = response.get("threshold_used")
            best_score = response.get("best_score")
        else:
            results, decision, threshold, best_score = [], "unknown", None, None

        # Бейдж
        if decision == "match":
            self.decision_badge.setText("MATCH")
            self.decision_badge.setObjectName("badgeMatch")
        else:
            self.decision_badge.setText("UNKNOWN")
            self.decision_badge.setObjectName("badgeUnknown")
        # обновить стили после смены objectName
        self.decision_badge.style().unpolish(self.decision_badge)
        self.decision_badge.style().polish(self.decision_badge)

        self.info_score.set_value(f"{best_score:.4f}" if best_score is not None else "—")
        self.info_threshold.set_value(f"{threshold:.4f}" if threshold is not None else "—")
        self.info_latency.set_value(f"{latency_ms:.0f} ms")
        self.info_matches.set_value(str(len(results)))

        # Таблица
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(results))
        for row_idx, result in enumerate(results):
            self.table.setItem(row_idx, 0, QTableWidgetItem(result.get("person_id", "")))
            self.table.setItem(row_idx, 1, QTableWidgetItem(result.get("label") or ""))
            self.table.setItem(row_idx, 2, NumericItem(result.get("score")))
            self.table.setItem(row_idx, 3, NumericItem(result.get("distance")))
        self.table.setSortingEnabled(True)
        self.table.sortItems(2, Qt.DescendingOrder)

    def _on_error(self, error: str) -> None:
        self.search_btn.setEnabled(True)
        QMessageBox.critical(self, "Search failed", error)
