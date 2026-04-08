from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency check
    cv2 = None

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.api_client import ApiClient
from app.core.config import DesktopSettings
from app.core.worker import ApiWorker
from app.ui.activity import record_event
from app.ui.dialogs import show_error, show_warning
from app.ui.widgets import (
    ActionButton,
    Card,
    CollapsibleSection,
    ConsoleView,
    DimLabel,
    ImageDropZone,
    InfoRow,
    LiveFaceLine,
    SectionHeading,
    StatusPill,
    shorten_path,
)


class SearchTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.settings = DesktopSettings()
        self.api = ApiClient(self.settings)
        self._worker: ApiWorker | None = None
        self._request_started = 0.0
        self._mode = "search"
        self._request_origin = "manual"
        self._camera = None
        self._live_running = False
        self._live_paused = False
        self._latest_frame: Any | None = None
        self._frame_for_request: Any | None = None
        self._live_annotations: list[dict[str, Any]] = []
        self._camera_timer = QTimer(self)
        self._camera_timer.timeout.connect(self._update_camera_frame)
        self._live_timer = QTimer(self)
        self._live_timer.timeout.connect(self._submit_live_frame)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(18)

        header = QVBoxLayout()
        header.setSpacing(4)
        title = QLabel("Face search")
        title.setObjectName("brandTitle")
        subtitle = DimLabel("Choose an image or camera feed, run a search, then inspect the matches.")
        header.addWidget(title)
        header.addWidget(subtitle)
        root.addLayout(header)

        body_row = QHBoxLayout()
        body_row.setSpacing(18)

        left_card = Card()
        left = left_card.body()
        left.addWidget(SectionHeading("Search workflow"))

        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_search_btn = self._make_mode_button("Search", "search")
        self.mode_enroll_btn = self._make_mode_button("Enroll", "enroll")
        self.mode_compare_btn = self._make_mode_button("Compare", "compare")
        mode_row.addWidget(self.mode_search_btn)
        mode_row.addWidget(self.mode_enroll_btn)
        mode_row.addWidget(self.mode_compare_btn)
        mode_row.addStretch()
        left.addLayout(mode_row)

        request_row = QHBoxLayout()
        request_row.setSpacing(10)
        self.pipeline_combo = QComboBox()
        self.pipeline_combo.setMinimumWidth(150)
        self.k_input = QSpinBox()
        self.k_input.setRange(1, 20)
        self.k_input.setValue(5)
        self.k_input.setPrefix("Top K  ")
        self.k_input.setMinimumWidth(110)
        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("Person label")
        request_row.addWidget(self.pipeline_combo)
        request_row.addWidget(self.k_input)
        request_row.addWidget(self.label_input, 1)
        left.addLayout(request_row)

        self.drop_zone = ImageDropZone()
        self.drop_zone.fileDropped.connect(self._set_image_path)
        left.addWidget(self.drop_zone, 1)

        self.file_label = DimLabel("No image selected")
        left.addWidget(self.file_label)

        image_actions = QHBoxLayout()
        image_actions.setSpacing(10)
        self.browse_btn = ActionButton("Browse image")
        self.browse_btn.clicked.connect(self._browse)
        self.clear_btn = ActionButton("Clear")
        self.clear_btn.clicked.connect(self._clear_image)
        self.capture_btn = ActionButton("Start camera")
        self.capture_btn.clicked.connect(self._toggle_camera)
        image_actions.addWidget(self.browse_btn)
        image_actions.addWidget(self.clear_btn)
        image_actions.addWidget(self.capture_btn)
        image_actions.addStretch()
        left.addLayout(image_actions)

        self.execute_btn = ActionButton("Run search", primary=True)
        self.execute_btn.clicked.connect(self._execute)
        left.addWidget(self.execute_btn)

        self.advanced_section = CollapsibleSection("Advanced options", expanded=False)
        advanced = self.advanced_section.body()
        advanced.addWidget(DimLabel("Live camera settings stay here so the main search path remains clean."))
        advanced_row = QHBoxLayout()
        advanced_row.setSpacing(10)
        self.live_status = StatusPill("Camera off", state="idle")
        self.camera_combo = QComboBox()
        for idx in range(5):
            self.camera_combo.addItem(f"Camera {idx}", idx)
        self.camera_combo.setCurrentIndex(min(max(self.settings.camera_index, 0), 4))
        self.live_preset = QComboBox()
        self.live_preset.addItem("Fast", "fast")
        self.live_preset.addItem("Balanced", "balanced")
        self.live_preset.addItem("Safe", "safe")
        self.live_preset.addItem("Custom", "custom")
        self.live_preset.setCurrentIndex(1)
        self.live_preset.currentIndexChanged.connect(self._on_live_preset_changed)
        self.live_interval = QSpinBox()
        self.live_interval.setRange(500, 5000)
        self.live_interval.setSingleStep(100)
        self.live_interval.setValue(self.settings.live_scan_interval_ms)
        self.live_interval.setSuffix(" ms")
        self.pause_btn = ActionButton("Pause scan")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._toggle_pause)
        advanced_row.addWidget(self.live_status)
        advanced_row.addWidget(self.camera_combo)
        advanced_row.addWidget(self.live_preset)
        advanced_row.addWidget(self.live_interval)
        advanced_row.addWidget(self.pause_btn)
        advanced_row.addStretch()
        advanced.addLayout(advanced_row)
        self.live_interval.valueChanged.connect(self._update_live_interval)
        left.addWidget(self.advanced_section)

        body_row.addWidget(left_card, 3)

        right_col = QVBoxLayout()
        right_col.setSpacing(16)

        summary_card = Card()
        summary = summary_card.body()
        summary_header = QHBoxLayout()
        summary_header.addWidget(SectionHeading("Result summary"))
        summary_header.addStretch()
        self.matches_pill = StatusPill("No results", state="idle")
        summary_header.addWidget(self.matches_pill)
        summary.addLayout(summary_header)

        summary_grid = QGridLayout()
        summary_grid.setHorizontalSpacing(18)
        summary_grid.setVerticalSpacing(12)
        self.info_mode = InfoRow("Mode", "-")
        self.info_best = InfoRow("Best score", "-")
        self.info_latency = InfoRow("Latency", "-")
        self.info_faces = InfoRow("Matches", "-")
        summary_grid.addWidget(self.info_mode, 0, 0)
        summary_grid.addWidget(self.info_best, 0, 1)
        summary_grid.addWidget(self.info_latency, 1, 0)
        summary_grid.addWidget(self.info_faces, 1, 1)
        summary.addLayout(summary_grid)
        right_col.addWidget(summary_card)

        self.detected_faces_card = Card(variant="subtle")
        self.detected_faces_card.setVisible(False)
        detected = self.detected_faces_card.body()
        detected.addWidget(SectionHeading("Detected faces"))
        self.frame_faces_layout = QVBoxLayout()
        self.frame_faces_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_faces_layout.setSpacing(8)
        detected.addLayout(self.frame_faces_layout)
        right_col.addWidget(self.detected_faces_card)

        matches_card = Card()
        matches = matches_card.body()
        matches.addWidget(SectionHeading("Top matches"))
        self.matches_table = QTableWidget(0, 6)
        self.matches_table.setHorizontalHeaderLabels(["Label", "Person ID", "Pipeline", "Face", "Score", "Distance"])
        self.matches_table.verticalHeader().setVisible(False)
        self.matches_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.matches_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.matches_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.matches_table.setAlternatingRowColors(True)
        self.matches_table.horizontalHeader().setStretchLastSection(True)
        self.matches_table.setMinimumHeight(340)
        matches.addWidget(self.matches_table, 1)
        self.results_hint = DimLabel("Run a search to see matches here.")
        matches.addWidget(self.results_hint)
        right_col.addWidget(matches_card, 1)

        body_row.addLayout(right_col, 2)
        root.addLayout(body_row, 1)

        self.details_section = CollapsibleSection("Technical details", expanded=False)
        details = self.details_section.body()
        details.addWidget(DimLabel("Raw backend response is kept here so it does not compete with the main workflow."))
        self.console = ConsoleView("Backend payloads will appear here.")
        self.console.setMinimumHeight(180)
        details.addWidget(self.console)
        root.addWidget(self.details_section)

        self._set_mode("search")

    def _make_mode_button(self, text: str, mode: str) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("toolbarButton")
        button.setCheckable(True)
        button.clicked.connect(lambda: self._set_mode(mode))
        self.mode_group.addButton(button)
        return button

    def _set_mode(self, mode: str) -> None:
        if mode == "enroll" and self._live_running:
            self._stop_camera()

        self._mode = mode
        mapping = {
            "search": self.mode_search_btn,
            "enroll": self.mode_enroll_btn,
            "compare": self.mode_compare_btn,
        }
        mapping[mode].setChecked(True)

        self.label_input.setVisible(mode == "enroll")
        self.k_input.setVisible(mode != "enroll")
        self.capture_btn.setEnabled(mode != "enroll")
        self.advanced_section.setVisible(mode != "enroll")
        self.pause_btn.setEnabled(mode != "enroll" and self._live_running)

        self.pipeline_combo.clear()
        if mode == "compare":
            self.pipeline_combo.addItem("Dual pipeline", "compare")
            self.pipeline_combo.setEnabled(False)
            self.execute_btn.setText("Compare models")
        elif mode == "enroll":
            self.pipeline_combo.addItem("Pretrained", "pretrained")
            self.pipeline_combo.addItem("Custom", "custom")
            self.pipeline_combo.addItem("Both", "both")
            self.pipeline_combo.setEnabled(True)
            self.execute_btn.setText("Enroll person")
        else:
            self.pipeline_combo.addItem("Pretrained", "pretrained")
            self.pipeline_combo.addItem("Custom", "custom")
            self.pipeline_combo.setEnabled(True)
            self.execute_btn.setText("Run search")

        if self.live_preset.currentData() != "custom":
            self._on_live_preset_changed()

        self.info_mode.set_value(mode.title())

    def _set_image_path(self, path: str) -> None:
        self.drop_zone.load(path)
        self.file_label.setText(shorten_path(path, max_length=74))
        self.file_label.setToolTip(path)
        self.file_label.setProperty("selected_path", path)
        if self._live_running:
            self._stop_camera()

    def _image_path(self) -> str:
        return str(self.file_label.property("selected_path") or "")

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All files (*.*)",
        )
        if path:
            self._set_image_path(path)

    def _clear_image(self) -> None:
        if self._live_running:
            self._stop_camera()
        self.drop_zone.clear_preview()
        self.file_label.setText("No image selected")
        self.file_label.setToolTip("")
        self.file_label.setProperty("selected_path", "")
        self.console.clear()
        self._clear_results()
        self._clear_frame_faces()
        self.matches_pill.set_state("idle", "No results")
        self.info_best.set_value("-")
        self.info_latency.set_value("-")
        self.info_faces.set_value("-")

    def _execute(self) -> None:
        path = self._image_path()
        if not path:
            show_warning(self, "Missing image", "Select or drop an image first.")
            return

        if self._mode == "enroll" and not self.label_input.text().strip():
            show_warning(self, "Missing label", "Enroll mode expects a profile label.")
            return

        self._dispatch_request(path, origin="manual", source_label=path)

    def _dispatch_request(self, image_source: str | bytes, *, origin: str, source_label: str) -> None:
        if self._worker is not None and self._worker.isRunning():
            return

        self._request_origin = origin
        self._request_started = time.perf_counter()
        self.execute_btn.setEnabled(origin != "manual")

        if origin == "live":
            self.live_status.set_state("warn", "Scanning")
        else:
            record_event("search", f"Started {self._mode} request", severity="INFO", meta={"path": source_label})

        if self._mode == "compare":
            self._worker = ApiWorker(self.api.search_compare, image_source, self.k_input.value(), parent=self)
        elif self._mode == "enroll":
            self._worker = ApiWorker(
                self.api.enroll,
                image_source,
                self.label_input.text().strip(),
                self.pipeline_combo.currentData(),
                parent=self,
            )
        else:
            self._worker = ApiWorker(
                self.api.search,
                image_source,
                self.k_input.value(),
                self.pipeline_combo.currentData(),
                parent=self,
            )

        self._worker.finished.connect(self._on_success)
        self._worker.failed.connect(self._on_error)
        self._worker.start()

    def _on_success(self, payload: object) -> None:
        origin = self._request_origin
        self._worker = None
        self.execute_btn.setEnabled(True)
        latency_ms = (time.perf_counter() - self._request_started) * 1000
        if origin != "live" or self.details_section.is_expanded():
            self.console.setPlainText(json.dumps(payload, indent=2, ensure_ascii=False, default=str))

        if self._mode == "enroll":
            self._render_enroll(payload, latency_ms)
        elif self._mode == "compare":
            self._render_compare(payload, latency_ms)
        else:
            self._render_search(payload, latency_ms)

        if origin == "live":
            if self._live_paused:
                self.live_status.set_state("warn", f"Paused · {latency_ms:.0f} ms")
            else:
                self.live_status.set_state("ok", f"Live · {latency_ms:.0f} ms")
            self._update_live_annotations(payload)

    def _render_enroll(self, payload: object, latency_ms: float) -> None:
        self._clear_results()
        if not isinstance(payload, dict):
            return

        self.matches_pill.set_state("ok", "Profile registered")
        self.info_mode.set_value(f"Enroll · {str(payload.get('pipeline', '-')).title()}")
        self.info_best.set_value(str(payload.get("person_id", "-"))[:18])
        self.info_latency.set_value(f"{latency_ms:.0f} ms")
        self.info_faces.set_value(str(payload.get("faces_detected", 0)))

        row = {
            "label": self.label_input.text().strip() or "Registered",
            "person_id": payload.get("person_id", ""),
            "pipeline": payload.get("pipeline", ""),
            "face_index": 0,
            "score": 1.0,
            "distance": 0.0,
        }
        self._populate_results([row])
        record_event(
            "enroll",
            f"Registered {self.label_input.text().strip() or 'profile'}",
            severity="INFO",
            details=f"person_id={payload.get('person_id', '-')}",
        )

    def _render_search(self, payload: object, latency_ms: float) -> None:
        self._clear_results()
        if not isinstance(payload, dict):
            return

        results = payload.get("results", [])
        best_score = payload.get("best_score")
        matched_faces = payload.get("matched_faces", 0)
        faces_detected = payload.get("faces_detected", 0)
        decision = payload.get("decision", "unknown")

        pill_state = "ok" if decision == "match" else "warn"
        self.matches_pill.set_state(pill_state, f"{len(results)} matches")
        self.info_mode.set_value(str(payload.get("pipeline", "-")).title())
        self.info_best.set_value(f"{best_score:.4f}" if best_score is not None else "-")
        self.info_latency.set_value(f"{payload.get('latency_ms', latency_ms):.0f} ms")
        self.info_faces.set_value(f"{matched_faces}/{faces_detected}")

        self._populate_results(results, empty_text="No matches returned for this query.")
        if self._request_origin != "live":
            record_event(
                "search",
                f"Search finished with {len(results)} results",
                severity="INFO",
                details=f"decision={decision} best_score={best_score}",
            )

    def _render_compare(self, payload: object, latency_ms: float) -> None:
        self._clear_results()
        if not isinstance(payload, dict):
            return

        comparisons = payload.get("comparisons", [])
        fastest = payload.get("fastest_pipeline", "-")
        merged_results: list[dict] = []
        best_summary: list[str] = []
        total_faces = 0

        for item in comparisons:
            pipeline = item.get("pipeline", "?")
            best = item.get("best_score")
            best_summary.append(f"{pipeline}: {best:.4f}" if best is not None else f"{pipeline}: -")
            total_faces += int(item.get("faces_detected", 0))
            for result in item.get("results", []):
                row = dict(result)
                row["pipeline"] = pipeline
                merged_results.append(row)

        self.matches_pill.set_state("warn", f"{len(merged_results)} matches")
        self.info_mode.set_value(f"Compare · fastest {str(fastest).title()}")
        self.info_best.set_value(" | ".join(best_summary) if best_summary else "-")
        self.info_latency.set_value(f"{latency_ms:.0f} ms")
        self.info_faces.set_value(str(total_faces))

        merged_results.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        self._populate_results(merged_results, empty_text="Compare completed, but no matches were returned.")
        if self._request_origin != "live":
            record_event(
                "compare",
                f"Compared pipelines, fastest={fastest}",
                severity="INFO",
                details=", ".join(best_summary),
            )

    def _clear_results(self) -> None:
        self.matches_table.setRowCount(0)
        self.results_hint.setText("Run a search to see matches here.")

    def _populate_results(self, results: list[dict[str, Any]], *, empty_text: str | None = None) -> None:
        self.matches_table.setRowCount(len(results))
        if not results:
            self.results_hint.setText(empty_text or "No results.")
            return

        for row, result in enumerate(results):
            values = [
                str(result.get("label") or "Unknown"),
                str(result.get("person_id") or "")[:18],
                str(result.get("pipeline") or "-"),
                str(int(result.get("face_index", 0)) + 1),
                self._format_score(result.get("score")),
                self._format_score(result.get("distance")),
            ]
            for col, value in enumerate(values):
                self.matches_table.setItem(row, col, QTableWidgetItem(value))
        self.matches_table.resizeColumnsToContents()
        self.results_hint.setText(f"Showing {len(results)} result(s).")

    def _format_score(self, value: Any) -> str:
        if value is None or value == "":
            return "-"
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)

    def _on_error(self, error: str) -> None:
        origin = self._request_origin
        self._worker = None
        self.execute_btn.setEnabled(True)
        self.matches_pill.set_state("error", "Request failed")
        record_event("search", "Request failed", severity="ERROR", details=error)
        if origin != "live" or self.details_section.is_expanded():
            self.console.setPlainText(error)
        if origin == "live":
            self.live_status.set_state("error", "Live error")
            return
        show_error(self, "Operation failed", error)

    def apply_global_filter(self, text: str) -> None:
        query = text.strip().lower()
        for row in range(self.matches_table.rowCount()):
            haystack_parts = []
            for col in range(self.matches_table.columnCount()):
                item = self.matches_table.item(row, col)
                haystack_parts.append(item.text().lower() if item is not None else "")
            self.matches_table.setRowHidden(row, bool(query) and query not in " ".join(haystack_parts))

    def _update_live_interval(self, value: int) -> None:
        if self.live_preset.currentData() != "custom":
            self.live_preset.blockSignals(True)
            self.live_preset.setCurrentIndex(3)
            self.live_preset.blockSignals(False)
        if self._live_timer.isActive():
            self._live_timer.start(value)

    def _on_live_preset_changed(self) -> None:
        preset = self.live_preset.currentData()
        values = {
            "fast": 800 if self._mode != "compare" else 1200,
            "balanced": 1200 if self._mode != "compare" else 1800,
            "safe": 2000 if self._mode != "compare" else 2600,
        }
        if preset in values:
            self.live_interval.blockSignals(True)
            self.live_interval.setValue(values[preset])
            self.live_interval.blockSignals(False)
            if self._live_timer.isActive():
                self._live_timer.start(self.live_interval.value())

    def _toggle_camera(self) -> None:
        if self._live_running:
            self._stop_camera()
            return
        self._start_camera()

    def _toggle_pause(self) -> None:
        if not self._live_running:
            return
        if self._live_paused:
            self._resume_live_scan()
        else:
            self._pause_live_scan()

    def _start_camera(self) -> None:
        if self._mode == "enroll":
            show_warning(self, "Camera unavailable", "Live camera mode is available only for Search or Compare.")
            return
        if cv2 is None:
            show_error(self, "Camera unavailable", "OpenCV is not installed in the desktop environment.")
            return

        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if hasattr(cv2, "CAP_DSHOW") else [cv2.CAP_ANY]
        camera = None
        selected_index = int(self.camera_combo.currentData())
        for backend in backends:
            candidate = cv2.VideoCapture(selected_index, backend)
            if candidate.isOpened():
                camera = candidate
                break
            candidate.release()
        if camera is None:
            show_error(self, "Camera unavailable", "Could not open the selected webcam.")
            return

        self._camera = camera
        self._latest_frame = None
        self._frame_for_request = None
        self._live_annotations = []
        self._live_running = True
        self._live_paused = False
        self.capture_btn.setText("Stop camera")
        self.browse_btn.setEnabled(False)
        self.camera_combo.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("Pause scan")
        self.live_status.set_state("ok", "Live camera")
        self._camera_timer.start(self.settings.camera_preview_interval_ms)
        self._live_timer.start(self.live_interval.value())
        record_event(
            "search",
            f"Started live camera {self._mode}",
            severity="INFO",
            details=f"camera={selected_index} interval={self.live_interval.value()}",
        )

    def _stop_camera(self) -> None:
        self._live_timer.stop()
        self._camera_timer.stop()
        self._live_running = False
        self._live_paused = False
        self._latest_frame = None
        self._frame_for_request = None
        self._live_annotations = []
        if self._camera is not None:
            self._camera.release()
            self._camera = None
        self.capture_btn.setText("Start camera")
        self.browse_btn.setEnabled(True)
        self.camera_combo.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("Pause scan")
        self.live_status.set_state("idle", "Camera off")
        self.drop_zone.clear_preview()
        self._clear_frame_faces()
        record_event("search", "Stopped live camera search", severity="INFO")

    def _pause_live_scan(self) -> None:
        self._live_paused = True
        self._live_timer.stop()
        self.pause_btn.setText("Resume scan")
        self.live_status.set_state("warn", "Scan paused")
        record_event("search", "Paused live camera scan", severity="INFO")

    def _resume_live_scan(self) -> None:
        self._live_paused = False
        self._live_timer.start(self.live_interval.value())
        self.pause_btn.setText("Pause scan")
        self.live_status.set_state("ok", "Live camera")
        record_event("search", "Resumed live camera scan", severity="INFO")

    def _update_camera_frame(self) -> None:
        if self._camera is None:
            return
        ok, frame = self._camera.read()
        if not ok:
            self.live_status.set_state("error", "Camera read failed")
            return
        self._latest_frame = frame
        # Only copy for the API request if the previous one was consumed
        if self._frame_for_request is None:
            self._frame_for_request = frame.copy()
        display = frame if not self._live_annotations else self._draw_live_annotations(frame.copy())
        self._show_frame(display)

    def _submit_live_frame(self) -> None:
        if not self._live_running or self._frame_for_request is None:
            return
        if self._worker is not None and self._worker.isRunning():
            return
        frame = self._frame_for_request
        self._frame_for_request = None  # mark as consumed
        encoded = self._encode_frame(frame)
        if encoded is None:
            self.live_status.set_state("error", "Encode failed")
            return
        self._dispatch_request(encoded, origin="live", source_label="camera")

    def _encode_frame(self, frame: Any) -> bytes | None:
        if cv2 is None:
            return None
        h, w = frame.shape[:2]
        # Downscale large frames for faster encoding and network transfer
        max_width = max(160, self.settings.live_max_width)
        if w > max_width:
            scale = max_width / w
            frame = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        quality = min(max(self.settings.live_jpeg_quality, 40), 95)
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return None
        return buffer.tobytes()

    def _show_frame(self, frame: Any) -> None:
        pixmap = self._frame_to_pixmap(frame)
        if pixmap is not None:
            self.drop_zone.set_pixmap(pixmap)
            self.file_label.setText("Live camera")
            self.file_label.setToolTip("Webcam stream")
            self.file_label.setProperty("selected_path", "")

    def _frame_to_pixmap(self, frame: Any) -> QPixmap | None:
        if cv2 is None:
            return None
        # Downscale for display if the frame is very large
        h, w = frame.shape[:2]
        max_w = 640
        if w > max_w:
            scale = max_w / w
            frame = cv2.resize(frame, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(image)

    def _draw_live_annotations(self, frame: Any) -> Any:
        if cv2 is None or not self._live_annotations:
            return frame
        for annotation in self._live_annotations:
            bbox = annotation.get("face_bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(max(0, value)) for value in bbox]
            quality = annotation.get("quality", "unknown")
            color = self._overlay_color(quality)
            label = annotation.get("label") or "Unknown"
            score = annotation.get("score")
            pipeline = annotation.get("pipeline")
            parts = [label]
            if pipeline:
                parts.append(str(pipeline))
            if score is not None:
                parts.append(f"{float(score):.3f}")
            text = " | ".join(parts)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(
                frame,
                (x1, max(0, y1 - 26)),
                (min(frame.shape[1] - 1, x1 + max(120, len(text) * 7)), y1),
                color,
                -1,
            )
            cv2.putText(
                frame,
                text,
                (x1 + 4, max(14, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (4, 16, 23),
                1,
                cv2.LINE_AA,
            )
        return frame

    def _update_live_annotations(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        self._live_annotations = self._extract_annotations(payload)
        self._render_frame_faces(self._live_annotations)
        # No need to force-redraw here; the camera timer will pick up
        # the new annotations on the next frame tick (~33ms away)

    def _overlay_color(self, quality: str) -> tuple[int, int, int]:
        if quality == "match":
            return (55, 243, 187)
        if quality == "weak":
            return (115, 204, 255)
        return (136, 107, 255)

    def _extract_annotations(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        thresholds: dict[int, float] = {}

        if "comparisons" in payload:
            for comparison in payload.get("comparisons", []):
                pipeline = comparison.get("pipeline")
                threshold = float(comparison.get("threshold_used", 0.0))
                for result in comparison.get("results", []):
                    row = dict(result)
                    row["pipeline"] = pipeline
                    grouped[int(row.get("face_index", 0))].append(row)
                    thresholds[int(row.get("face_index", 0))] = max(
                        threshold,
                        thresholds.get(int(row.get("face_index", 0)), 0.0),
                    )
        else:
            threshold = float(payload.get("threshold_used", 0.0))
            for result in payload.get("results", []):
                grouped[int(result.get("face_index", 0))].append(dict(result))
                thresholds[int(result.get("face_index", 0))] = threshold

        annotations: list[dict[str, Any]] = []
        matched_face_indices: set[int] = set()

        for face_index, results in grouped.items():
            best = max(results, key=lambda item: float(item.get("score", 0.0)))
            best["face_index"] = face_index
            threshold = thresholds.get(face_index, 0.0)
            score = float(best.get("score", 0.0))
            if score >= threshold and threshold > 0:
                best["quality"] = "match"
            elif score >= max(0.2, threshold * 0.7):
                best["quality"] = "weak"
            else:
                best["quality"] = "unknown"
            annotations.append(best)
            matched_face_indices.add(face_index)

        detected_faces: list[dict[str, Any]] = []
        if "comparisons" in payload:
            for comparison in payload.get("comparisons", []):
                detected_faces.extend(comparison.get("detected_faces", []))
        else:
            detected_faces = payload.get("detected_faces", [])

        for face_info in detected_faces:
            face_index = int(face_info.get("face_index", 0))
            if face_index in matched_face_indices:
                continue
            annotations.append(
                {
                    "face_index": face_index,
                    "face_bbox": face_info.get("face_bbox"),
                    "detection_score": face_info.get("detection_score"),
                    "label": "No match",
                    "quality": "unknown",
                    "score": None,
                }
            )

        return sorted(annotations, key=lambda item: int(item.get("face_index", 0)))

    def _clear_frame_faces(self) -> None:
        while self.frame_faces_layout.count():
            item = self.frame_faces_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.detected_faces_card.setVisible(False)

    def _render_frame_faces(self, faces: list[dict[str, Any]]) -> None:
        while self.frame_faces_layout.count():
            item = self.frame_faces_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not faces:
            self.detected_faces_card.setVisible(False)
            return

        self.detected_faces_card.setVisible(True)
        for face in faces:
            widget = LiveFaceLine()
            widget.set_face(face)
            self.frame_faces_layout.addWidget(widget)

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        if self._live_running:
            self._stop_camera()
