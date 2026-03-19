# Главная вкладка — поиск лица по индексу
from __future__ import annotations

import json
import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.api_client import ApiClient
from app.core.worker import ApiWorker
from app.ui.dialogs import show_error, show_warning
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

        top_row = QHBoxLayout()
        top_row.setSpacing(16)

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

        controls_row = QHBoxLayout()
        controls_row.setSpacing(16)

        k_group = QHBoxLayout()
        k_group.setSpacing(8)
        k_group.addWidget(DimLabel("Top K"))
        self.k_input = QSpinBox()
        self.k_input.setRange(1, 100)
        self.k_input.setValue(5)
        self.k_input.setFixedWidth(70)
        k_group.addWidget(self.k_input)
        controls_row.addLayout(k_group)

        mode_group = QHBoxLayout()
        mode_group.setSpacing(8)
        mode_group.addWidget(DimLabel("Mode"))
        self.pipeline_combo = QComboBox()
        self.pipeline_combo.addItem("Pretrained", "pretrained")
        self.pipeline_combo.addItem("Custom", "custom")
        self.pipeline_combo.addItem("Compare both", "compare")
        self.pipeline_combo.setMinimumWidth(150)
        mode_group.addWidget(self.pipeline_combo)
        controls_row.addLayout(mode_group)

        controls_row.addStretch()
        fc.addLayout(controls_row)

        fc.addSpacing(10)

        self.search_btn = QPushButton("Search")
        self.search_btn.setObjectName("primary")
        self.search_btn.setFixedHeight(36)
        self.search_btn.clicked.connect(self._search)
        fc.addWidget(self.search_btn)
        fc.addStretch()

        top_row.addWidget(form_card, 3)

        right_col = QVBoxLayout()
        right_col.setSpacing(12)

        preview_card = Card()
        pc = preview_card.body()
        pc.setAlignment(Qt.AlignCenter)
        self.preview = ImagePreview(160)
        pc.addWidget(self.preview, 0, Qt.AlignCenter)
        right_col.addWidget(preview_card)

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

        summary_card = Card()
        sc = summary_card.body()
        info_row = QHBoxLayout()
        info_row.setSpacing(24)
        self.info_score = InfoRow("Best score")
        self.info_threshold = InfoRow("Threshold/Fastest")
        self.info_latency = InfoRow("Latency")
        self.info_matches = InfoRow("Matches")
        for widget in [self.info_score, self.info_threshold, self.info_latency, self.info_matches]:
            info_row.addWidget(widget)
        info_row.addStretch()
        sc.addLayout(info_row)
        root.addWidget(summary_card)

        results_card = Card()
        rc = results_card.body()
        rc.addWidget(SectionHeading("Search results"))

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Face", "Det. score", "Pipeline", "Person ID", "Label", "Score", "Distance"]
        )
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.Stretch)
        hdr.setSectionResizeMode(4, QHeaderView.Stretch)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        rc.addWidget(self.table)

        rc.addWidget(DimLabel("Compare details"))
        self.compare_view = QTextEdit()
        self.compare_view.setReadOnly(True)
        self.compare_view.setMaximumHeight(160)
        self.compare_view.setPlaceholderText("Compare mode will show both model latencies here.")
        rc.addWidget(self.compare_view)

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
            show_warning(self, "Missing", "Select an image file")
            return

        mode = self.pipeline_combo.currentData()
        k = self.k_input.value()
        self.search_btn.setEnabled(False)
        self._search_start = time.perf_counter()

        if mode == "compare":
            self._worker = ApiWorker(self.api.search_compare, path, k, parent=self)
        else:
            self._worker = ApiWorker(self.api.search, path, k, mode, parent=self)
        self._worker.finished.connect(self._on_success)
        self._worker.failed.connect(self._on_error)
        self._worker.start()

    def _set_badge(self, state: str) -> None:
        if state == "match":
            self.decision_badge.setText("MATCH")
            self.decision_badge.setObjectName("badgeMatch")
        elif state == "compare":
            self.decision_badge.setText("COMPARE")
            self.decision_badge.setObjectName("badgeCompare")
        else:
            self.decision_badge.setText("UNKNOWN")
            self.decision_badge.setObjectName("badgeUnknown")
        self.decision_badge.style().unpolish(self.decision_badge)
        self.decision_badge.style().polish(self.decision_badge)

    def _populate_rows(self, rows: list[dict]) -> None:
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rows))
        for row_idx, result in enumerate(rows):
            face_index = int(result.get("face_index", 0)) + 1
            self.table.setItem(row_idx, 0, QTableWidgetItem(str(face_index)))
            self.table.setItem(row_idx, 1, NumericItem(result.get("detection_score")))
            self.table.setItem(row_idx, 2, QTableWidgetItem(result.get("pipeline", "")))
            self.table.setItem(row_idx, 3, QTableWidgetItem(result.get("person_id", "")))
            self.table.setItem(row_idx, 4, QTableWidgetItem(result.get("label") or ""))
            self.table.setItem(row_idx, 5, NumericItem(result.get("score")))
            self.table.setItem(row_idx, 6, NumericItem(result.get("distance")))
        self.table.setSortingEnabled(True)
        self.table.sortItems(5, Qt.DescendingOrder)

    def _on_success(self, response: object) -> None:
        self.search_btn.setEnabled(True)
        request_latency_ms = (time.perf_counter() - self._search_start) * 1000

        if isinstance(response, dict) and "comparisons" in response:
            self._render_compare(response, request_latency_ms)
        else:
            self._render_single(response, request_latency_ms)

    def _render_single(self, response: object, request_latency_ms: float) -> None:
        if isinstance(response, dict):
            results = response.get("results", [])
            decision = response.get("decision", "unknown")
            threshold = response.get("threshold_used")
            best_score = response.get("best_score")
            pipeline = response.get("pipeline") or ""
            model_latency = response.get("latency_ms")
            faces_detected = int(response.get("faces_detected", 0))
            matched_faces = int(response.get("matched_faces", 0))
        else:
            results, decision, threshold, best_score, pipeline, model_latency = [], "unknown", None, None, "", None
            faces_detected = 0
            matched_faces = 0

        self._set_badge(decision)
        self.info_score.set_value(f"{best_score:.4f}" if best_score is not None else "—")
        self.info_threshold.set_value(f"{threshold:.4f}" if threshold is not None else "—")
        if model_latency is not None:
            self.info_latency.set_value(f"{model_latency:.0f} ms model / {request_latency_ms:.0f} ms req")
        else:
            self.info_latency.set_value(f"{request_latency_ms:.0f} ms")
        self.info_matches.set_value(f"{matched_faces}/{faces_detected} faces")

        rows = []
        for result in results:
            row = dict(result)
            row["pipeline"] = row.get("pipeline") or pipeline
            rows.append(row)
        self._populate_rows(rows)
        self.compare_view.setPlainText(
            json.dumps(
                {
                    "pipeline": pipeline,
                    "faces_detected": faces_detected,
                    "matched_faces": matched_faces,
                    "latency_ms": model_latency,
                    "threshold_used": threshold,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    def _render_compare(self, response: dict, request_latency_ms: float) -> None:
        comparisons = response.get("comparisons", [])
        rows: list[dict] = []
        summary: list[dict] = []

        best_parts: list[str] = []
        latency_parts: list[str] = []
        for item in comparisons:
            pipeline = item.get("pipeline", "?")
            best_score = item.get("best_score")
            latency_ms = item.get("latency_ms")
            best_parts.append(
                f"{pipeline}={best_score:.4f}" if best_score is not None else f"{pipeline}=—"
            )
            latency_parts.append(
                f"{pipeline}={latency_ms:.0f} ms" if latency_ms is not None else f"{pipeline}=—"
            )
            summary.append(
                {
                    "pipeline": pipeline,
                    "model": item.get("model"),
                    "latency_ms": latency_ms,
                    "decision": item.get("decision"),
                    "best_score": best_score,
                    "faces_detected": item.get("faces_detected"),
                    "matched_faces": item.get("matched_faces"),
                    "error": item.get("error"),
                    "top1": (item.get("results") or [{}])[0].get("label") if item.get("results") else None,
                }
            )
            for result in item.get("results", []):
                row = dict(result)
                row["pipeline"] = pipeline
                rows.append(row)

        self._set_badge("compare")
        self.info_score.set_value(" / ".join(best_parts) if best_parts else "—")
        self.info_threshold.set_value(response.get("fastest_pipeline") or "—")
        self.info_latency.set_value(" / ".join(latency_parts) if latency_parts else f"{request_latency_ms:.0f} ms")
        total_faces = sum(int(item.get("faces_detected", 0)) for item in comparisons)
        self.info_matches.set_value(f"{len(rows)} hits / {total_faces} faces")
        self._populate_rows(rows)
        self.compare_view.setPlainText(json.dumps(summary, indent=2, ensure_ascii=False))

    def _on_error(self, error: str) -> None:
        self.search_btn.setEnabled(True)
        show_error(self, "Search failed", error)
