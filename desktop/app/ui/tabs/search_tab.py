from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency check
    cv2 = None

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
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

from app.core.api_client import ApiClient, format_api_error
from app.core.config import DesktopSettings
from app.core.worker import ApiWorker
from app.ui.activity import record_event
from app.ui.dialogs import show_error, show_warning
from app.ui.frame_processing import (
    draw_live_annotations,
    encode_frame_for_upload,
    frame_to_pixmap,
)
from app.ui.live_geometry import scale_bbox
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
        self._request_bbox_scale: tuple[float, float] = (1.0, 1.0)
        self._selected_image_paths: list[str] = []
        self._live_annotations: list[dict[str, Any]] = []
        self._live_result_history: list[tuple[str, float] | None] = []
        self._live_stable_label: str | None = None
        self._live_stable_misses = 0
        self._live_checking_ticks = 0
        self._live_no_face_misses = 0
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
        mode_row.addWidget(self.mode_search_btn)
        mode_row.addWidget(self.mode_enroll_btn)
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
        self.live_interval.setRange(150, 5000)
        self.live_interval.setSingleStep(50)
        self.live_interval.setValue(self.settings.live_scan_interval_ms)
        self.live_interval.setSuffix(" ms")
        self.multi_face_live = QCheckBox("Multi-face search")
        self.multi_face_live.setChecked(False)
        self.pause_btn = ActionButton("Pause scan")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._toggle_pause)
        advanced_row.addWidget(self.live_status)
        advanced_row.addWidget(self.camera_combo)
        advanced_row.addWidget(self.live_preset)
        advanced_row.addWidget(self.live_interval)
        advanced_row.addWidget(self.multi_face_live)
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

        self.summary_message = QLabel("No search has been run yet.")
        self.summary_message.setObjectName("decisionText")
        self.summary_message.setWordWrap(True)
        self.summary_message.setProperty("state", "idle")
        summary.addWidget(self.summary_message)


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
        self.matches_table = QTableWidget(0, 5)
        self.matches_table.setHorizontalHeaderLabels(["Label", "Decision", "Pipeline", "Face", "Score"])
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
        }
        mapping[mode].setChecked(True)

        self.label_input.setVisible(mode == "enroll")
        self.k_input.setVisible(mode != "enroll")
        self.capture_btn.setEnabled(mode != "enroll")
        self.advanced_section.setVisible(mode != "enroll")
        self.pause_btn.setEnabled(mode != "enroll" and self._live_running)

        self.pipeline_combo.clear()
        if mode == "enroll":
            self.pipeline_combo.addItem("Both pipelines", "both")
            self.pipeline_combo.setEnabled(False)
            self.execute_btn.setText("Enroll person")
            self.browse_btn.setText("Browse images")
        else:
            if len(self._selected_image_paths) > 1:
                self._set_image_path(self._selected_image_paths[0])
            self.pipeline_combo.addItem("Pretrained", "pretrained")
            self.pipeline_combo.addItem("Custom", "custom")
            self.pipeline_combo.setEnabled(True)
            self.execute_btn.setText("Run search")
            self.browse_btn.setText("Browse image")

        if self.live_preset.currentData() != "custom":
            self._on_live_preset_changed()

        self.info_mode.set_value(mode.title())

    def _set_image_path(self, path: str) -> None:
        self._set_image_paths([path])

    def _set_image_paths(self, paths: list[str]) -> None:
        selected_paths = [path for path in paths if path]
        if not selected_paths:
            return

        first_path = selected_paths[0]
        self._selected_image_paths = selected_paths
        self.drop_zone.load(first_path)
        if len(selected_paths) == 1:
            self.file_label.setText(shorten_path(first_path, max_length=74))
        else:
            preview = shorten_path(first_path, max_length=52)
            self.file_label.setText(f"{len(selected_paths)} images selected. Preview: {preview}")
        self.file_label.setToolTip("\n".join(selected_paths[:20]))
        self.file_label.setProperty("selected_path", first_path)
        self.file_label.setProperty("selected_paths", selected_paths)
        if self._live_running:
            self._stop_camera()

    def _image_path(self) -> str:
        return str(self.file_label.property("selected_path") or "")

    def _image_paths(self) -> list[str]:
        if self._selected_image_paths:
            return list(self._selected_image_paths)
        path = self._image_path()
        return [path] if path else []

    def _browse(self) -> None:
        if self._mode == "enroll":
            paths, _ = QFileDialog.getOpenFileNames(
                self,
                "Select images",
                "",
                "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All files (*.*)",
            )
            if paths:
                self._set_image_paths(paths)
            return

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
        self._selected_image_paths = []
        self.drop_zone.clear_preview()
        self.file_label.setText("No image selected")
        self.file_label.setToolTip("")
        self.file_label.setProperty("selected_path", "")
        self.file_label.setProperty("selected_paths", [])
        self.console.clear()
        self._clear_results()
        self._clear_frame_faces()
        self.matches_pill.set_state("idle", "No results")
        self.info_best.set_value("-")
        self.info_latency.set_value("-")
        self.info_faces.set_value("-")
        self._set_summary_message("idle", "No search has been run yet.")

    def _execute(self) -> None:
        paths = self._image_paths()
        if not paths:
            show_warning(self, "Missing image", "Select or drop an image first.")
            return

        if self._mode == "enroll" and not self.label_input.text().strip():
            show_warning(self, "Missing label", "Enroll mode expects a profile label.")
            return

        if self._mode == "enroll" and len(paths) > 1:
            self._dispatch_enroll_batch(paths)
            return

        path = paths[0]
        self._dispatch_request(
            path,
            origin="manual",
            source_label=path,
            bbox_scale=(1.0, 1.0),
        )

    def _dispatch_request(
        self,
        image_source: str | bytes,
        *,
        origin: str,
        source_label: str,
        bbox_scale: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        if self._worker is not None and self._worker.isRunning():
            return

        self._request_origin = origin
        self._request_bbox_scale = bbox_scale
        self._request_started = time.perf_counter()
        self.execute_btn.setEnabled(origin != "manual")

        if origin == "live":
            self.live_status.set_state("warn", "Scanning")
        else:
            record_event("search", f"Started {self._mode} request", severity="INFO", meta={"path": source_label})

        if self._mode == "enroll":
            self._worker = ApiWorker(
                self.api.enroll,
                image_source,
                self.label_input.text().strip(),
                self.pipeline_combo.currentData(),
                parent=self,
            )
        else:
            request_k = 1 if origin == "live" else self.k_input.value()
            self._worker = ApiWorker(
                self.api.search,
                image_source,
                request_k,
                self.pipeline_combo.currentData(),
                source="webcam" if origin == "live" else None,
                multi_face=self.multi_face_live.isChecked() if origin == "live" else True,
                parent=self,
            )

        self._worker.finished.connect(self._on_success)
        self._worker.failed.connect(self._on_error)
        self._worker.start()

    def _dispatch_enroll_batch(self, image_paths: list[str]) -> None:
        if self._worker is not None and self._worker.isRunning():
            return

        label = self.label_input.text().strip()
        pipeline = str(self.pipeline_combo.currentData() or "both")
        self._request_origin = "manual"
        self._request_bbox_scale = (1.0, 1.0)
        self._request_started = time.perf_counter()
        self.execute_btn.setEnabled(False)
        record_event(
            "enroll",
            f"Started batch enroll for {len(image_paths)} images",
            severity="INFO",
            details=f"label={label} pipeline={pipeline}",
        )

        self._worker = ApiWorker(
            self._enroll_image_batch,
            list(image_paths),
            label,
            pipeline,
            parent=self,
        )
        self._worker.finished.connect(self._on_success)
        self._worker.failed.connect(self._on_error)
        self._worker.start()

    def _enroll_image_batch(self, image_paths: list[str], label: str, pipeline: str) -> dict[str, Any]:
        started = time.perf_counter()
        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for position, path in enumerate(image_paths, start=1):
            try:
                payload = self.api.enroll(path, label, pipeline)
            except Exception as exc:  # noqa: BLE001 - keep batch going and report per-file errors
                errors.append(
                    {
                        "position": position,
                        "file": Path(path).name,
                        "error": format_api_error(exc),
                    }
                )
                continue

            results.append(
                {
                    "position": position,
                    "file": Path(path).name,
                    "payload": payload,
                }
            )

        return {
            "operation": "batch_enroll",
            "label": label,
            "pipeline": pipeline,
            "requested": len(image_paths),
            "succeeded": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    def _on_success(self, payload: object) -> None:
        origin = self._request_origin
        self._worker = None
        self.execute_btn.setEnabled(True)
        latency_ms = (time.perf_counter() - self._request_started) * 1000
        if origin != "live" or self.details_section.is_expanded():
            self.console.setPlainText(json.dumps(payload, indent=2, ensure_ascii=False, default=str))

        if self._mode == "enroll":
            self._render_enroll(payload, latency_ms)
        else:
            self._render_search(payload, latency_ms)
            if origin != "live":
                self._update_manual_annotations(payload)

        if origin == "live":
            self._live_no_face_misses = 0
            if self._live_paused:
                self.live_status.set_state("warn", f"Paused - {latency_ms:.0f} ms")
            else:
                state, text = self._live_status_from_payload(payload, latency_ms)
                self.live_status.set_state(state, text)
            self._update_live_annotations(payload)

    def _render_enroll(self, payload: object, latency_ms: float) -> None:
        self._clear_results()
        if not isinstance(payload, dict):
            return
        if payload.get("operation") == "batch_enroll":
            self._render_batch_enroll(payload, latency_ms)
            return

        self.matches_pill.set_state("ok", "Profile registered")
        self.info_mode.set_value(f"Enroll - {str(payload.get('pipeline', '-')).title()}")
        self.info_best.set_value(str(payload.get("person_id", "-"))[:18])
        self.info_latency.set_value(f"{latency_ms:.0f} ms")
        self.info_faces.set_value(str(payload.get("faces_detected", 0)))
        self._set_summary_message("ok", "Profile registered and indexed for search.")


        row = {
            "label": self.label_input.text().strip() or "Registered",
            "person_id": payload.get("person_id", ""),
            "pipeline": payload.get("pipeline", ""),
            "face_index": 0,
            "score": 1.0,
            "distance": 0.0,
            "_decision": "Registered",
        }
        self._populate_results([row])
        record_event(
            "enroll",
            f"Registered {self.label_input.text().strip() or 'profile'}",
            severity="INFO",
            details=f"person_id={payload.get('person_id', '-')}",
        )

    def _render_batch_enroll(self, payload: dict[str, Any], latency_ms: float) -> None:
        requested = int(payload.get("requested", 0) or 0)
        succeeded = int(payload.get("succeeded", 0) or 0)
        failed = int(payload.get("failed", 0) or 0)
        pipeline = str(payload.get("pipeline") or "-")
        state = "ok" if failed == 0 and succeeded > 0 else "warn" if succeeded > 0 else "error"
        self.matches_pill.set_state(state, f"{succeeded}/{requested} registered")
        self.info_mode.set_value(f"Enroll batch - {pipeline.title()}")
        self.info_latency.set_value(f"{payload.get('latency_ms', latency_ms):.0f} ms")
        self.info_faces.set_value(f"{succeeded}/{requested}")

        rows: list[dict[str, Any]] = []
        first_person_id = "-"
        for row_index, item in enumerate(payload.get("results", [])):
            if not isinstance(item, dict):
                continue
            result_payload = item.get("payload")
            if not isinstance(result_payload, dict):
                result_payload = {}
            person_id = str(result_payload.get("person_id") or "")
            if first_person_id == "-" and person_id:
                first_person_id = person_id[:18]
            indexed_pipelines = result_payload.get("pipelines_indexed")
            if isinstance(indexed_pipelines, list) and indexed_pipelines:
                row_pipeline = ", ".join(str(value) for value in indexed_pipelines)
            else:
                row_pipeline = str(result_payload.get("pipeline") or pipeline)
            rows.append(
                {
                    "label": self.label_input.text().strip() or "Registered",
                    "person_id": person_id,
                    "pipeline": row_pipeline,
                    "face_index": row_index,
                    "score": 1.0,
                    "distance": 0.0,
                    "source": item.get("file", ""),
                    "_decision": "Registered",
                }
            )

        self.info_best.set_value(first_person_id)
        if succeeded:
            self._populate_results(rows, empty_text="No images were registered.")
        else:
            self._populate_results([], empty_text="No images were registered.")

        if failed:
            failed_files = payload.get("errors", [])
            first_error = ""
            if failed_files and isinstance(failed_files[0], dict):
                first_error = str(failed_files[0].get("error") or "")
            suffix = f" First error: {first_error}" if first_error else ""
            self._set_summary_message(
                state,
                f"Registered {succeeded}/{requested} images for this profile; {failed} failed validation.{suffix}",
            )
        else:
            self._set_summary_message("ok", f"Registered {succeeded} images as samples for the same profile.")

        record_event(
            "enroll",
            f"Batch enroll finished: {succeeded}/{requested} images registered",
            severity="INFO" if failed == 0 else "WARN",
            details=f"failed={failed} label={self.label_input.text().strip() or '-'}",
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
        display_results = results
        if self._request_origin == "live" and decision != "match":
            display_results = []

        pill_state = "ok" if decision == "match" else "warn"
        result_label = "matches" if decision == "match" else "matches"
        self.matches_pill.set_state(pill_state, f"{len(display_results)} {result_label}")
        self.info_mode.set_value(str(payload.get("pipeline", "-")).title())
        self.info_best.set_value(f"{best_score:.4f}" if best_score is not None else "-")
        self.info_latency.set_value(f"{payload.get('latency_ms', latency_ms):.0f} ms")
        self.info_faces.set_value(f"{matched_faces}/{faces_detected}")

        self._set_summary_message(
            pill_state,
            self._search_summary_text(decision, best_score, matched_faces, faces_detected),
        )

        self._populate_results(
            display_results,
            empty_text="No matches returned for this query.",
            threshold=payload.get("threshold_used"),
        )
        if self._request_origin != "live":
            record_event(
                "search",
                f"Search finished with {len(display_results)} results",
                severity="INFO",
                details=f"decision={decision} best_score={best_score}",
            )



    def _clear_results(self) -> None:
        self.matches_table.setRowCount(0)
        self.results_hint.setText("Run a search to see matches here.")

    def _set_summary_message(self, state: str, text: str) -> None:
        self.summary_message.setText(text)
        self.summary_message.setProperty("state", state)
        self.summary_message.style().unpolish(self.summary_message)
        self.summary_message.style().polish(self.summary_message)

    def _search_summary_text(
        self,
        decision: object,
        best_score: object,
        matched_faces: object,
        faces_detected: object,
    ) -> str:
        if decision == "match":
            score = self._format_score(best_score)
            return f"Match found. {matched_faces}/{faces_detected} detected face(s) passed the threshold. Best score: {score}."
        if int(faces_detected or 0) > 0:
            return f"No confident match. {faces_detected} face(s) detected, but scores stayed below the threshold."
        return "No face result is available for this request."



    def _populate_results(
        self,
        results: list[dict[str, Any]],
        *,
        empty_text: str | None = None,
        threshold: Any | None = None,
    ) -> None:
        self.matches_table.setRowCount(len(results))
        if not results:
            self.results_hint.setText(empty_text or "No results.")
            return

        for row, result in enumerate(results):
            values = [
                str(result.get("label") or "Unknown"),
                self._result_decision(result, threshold),
                str(result.get("pipeline") or "-"),
                str(int(result.get("face_index", 0)) + 1),
                self._format_score(result.get("score")),
            ]
            tooltip = self._result_tooltip(result)
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(tooltip)
                self.matches_table.setItem(row, col, item)
        self.matches_table.resizeColumnsToContents()
        self.results_hint.setText(f"Showing {len(results)} result(s).")

    def _result_decision(self, result: dict[str, Any], threshold: Any | None) -> str:
        explicit = result.get("_decision")
        if explicit:
            return str(explicit)

        score = result.get("score")
        if score is None:
            return "-"
        local_threshold = result.get("threshold_used", threshold)
        try:
            score_value = float(score)
            if local_threshold is not None:
                return "Match" if score_value >= float(local_threshold) else "Candidate"
        except (TypeError, ValueError):
            return "-"
        return "Candidate"

    def _result_tooltip(self, result: dict[str, Any]) -> str:
        parts = []
        for label, key in [
            ("File", "source"),
            ("Person ID", "person_id"),
            ("Embedding ID", "embedding_id"),
            ("Distance", "distance"),
            ("Detection", "detection_score"),
        ]:
            value = result.get(key)
            if value is not None and value != "":
                parts.append(f"{label}: {value}")
        return "\n".join(parts) if parts else "No additional details"

    def _live_status_from_payload(self, payload: object, latency_ms: float) -> tuple[str, str]:
        if not isinstance(payload, dict):
            return "ok", f"Live - {latency_ms:.0f} ms"


        decision = str(payload.get("decision", "")).lower()
        if decision == "match":
            return "ok", f"Match - {latency_ms:.0f} ms"
        if decision == "unknown":
            return "warn", f"Unknown - {latency_ms:.0f} ms"
        return "ok", f"Live - {latency_ms:.0f} ms"

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

        if origin == "live":
            self._handle_live_error(error)
            return

        self.matches_pill.set_state("error", "Request failed")
        self._set_summary_message("error", error)
        record_event("search", "Request failed", severity="ERROR", details=error)
        if origin != "live" or self.details_section.is_expanded():
            self.console.setPlainText(error)
        show_error(self, "Operation failed", error)

    def _handle_live_error(self, error: str) -> None:
        record_event("search", "Live request failed", severity="WARN", details=error)
        if self.details_section.is_expanded():
            self.console.setPlainText(error)

        if self._is_no_face_error(error):
            self._live_result_history.append(None)
            self._live_result_history = self._live_result_history[-3:]
            self._live_no_face_misses += 1

            if self._live_no_face_misses < 3 and self._live_annotations:
                self.live_status.set_state("warn", "Tracking")
                return

            if self._live_no_face_misses >= 3:
                self._live_annotations = []
                self._render_frame_faces([])
                self.matches_pill.set_state("warn", "No face")
                self.info_best.set_value("-")
                self.info_faces.set_value("0/0")
                self._set_summary_message("warn", "No face detected in the last live frames.")
                self.live_status.set_state("warn", "No face")
                return

            self.live_status.set_state("warn", "Looking for face")
            return

        self.matches_pill.set_state("error", "Request failed")
        self._set_summary_message("error", error)
        self.live_status.set_state("error", "Live error")

    def _is_no_face_error(self, error: str) -> bool:
        normalized = error.lower()
        return (
            "no face" in normalized
            or "face detected" in normalized
            or "face was detected" in normalized
            or "no faces" in normalized
        )

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
            "fast": 150,
            "balanced": 300,
            "safe": 600,
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
            show_warning(self, "Camera unavailable", "Live camera mode is available only for Search.")
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

        if self.settings.camera_frame_width > 0:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.camera_frame_width)
        if self.settings.camera_frame_height > 0:
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.camera_frame_height)

        self._camera = camera
        self._latest_frame = None
        self._frame_for_request = None
        self._request_bbox_scale = (1.0, 1.0)
        self._live_annotations = []
        self._live_result_history = []
        self._live_stable_label = None
        self._live_stable_misses = 0
        self._live_checking_ticks = 0
        self._live_no_face_misses = 0
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
        self._live_result_history = []
        self._live_no_face_misses = 0
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
        display = frame if not self._live_annotations else draw_live_annotations(
            frame.copy(),
            self._live_annotations,
        )
        self._show_frame(display)

    def _submit_live_frame(self) -> None:
        if not self._live_running or self._frame_for_request is None:
            return
        if self._worker is not None and self._worker.isRunning():
            return
        frame = self._frame_for_request
        self._frame_for_request = None  # mark as consumed
        max_width, jpeg_quality = self._live_upload_settings()
        encoded_result = encode_frame_for_upload(
            frame,
            max_width=max_width,
            jpeg_quality=jpeg_quality,
        )
        if encoded_result is None:
            self.live_status.set_state("error", "Encode failed")
            return
        encoded, bbox_scale = encoded_result
        self._dispatch_request(
            encoded,
            origin="live",
            source_label="camera",
            bbox_scale=bbox_scale,
        )

    def _live_upload_settings(self) -> tuple[int, int]:
        pipeline = str(self.pipeline_combo.currentData() or "")
        if pipeline == "custom":
            return (
                max(self.settings.live_max_width, self.settings.custom_live_max_width),
                max(self.settings.live_jpeg_quality, self.settings.custom_live_jpeg_quality),
            )
        return min(self.settings.live_max_width, 640), min(self.settings.live_jpeg_quality, 75)

    def _show_frame(self, frame: Any) -> None:
        pixmap = frame_to_pixmap(frame)
        if pixmap is not None:
            self._selected_image_paths = []
            self.drop_zone.set_pixmap(pixmap)
            self.file_label.setText("Live camera")
            self.file_label.setToolTip("Webcam stream")
            self.file_label.setProperty("selected_path", "")
            self.file_label.setProperty("selected_paths", [])

    def _update_live_annotations(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        self._live_no_face_misses = 0
        annotations = self._extract_annotations(payload, self._request_bbox_scale)
        if (
            not annotations
            and str(payload.get("decision", "")).lower() != "match"
        ):
            annotations = []
        elif not annotations and self._live_annotations:
            annotations = self._live_annotations

        for annotation in annotations:
            if annotation.get("quality") != "match":
                annotation["label"] = "No match"
        self._live_annotations = annotations
        self._render_frame_faces(self._live_annotations)
        # No need to force-redraw here; the camera timer will pick up
        # the new annotations on the next frame tick (~33ms away)

    def _update_manual_annotations(self, payload: object) -> None:
        if cv2 is None or not isinstance(payload, dict):
            return
        paths = self._image_paths()
        if len(paths) != 1:
            return
        frame = cv2.imread(paths[0], cv2.IMREAD_COLOR)
        if frame is None:
            return

        annotations = self._extract_annotations(payload, self._request_bbox_scale)
        if annotations:
            frame = draw_live_annotations(frame, annotations)
        pixmap = frame_to_pixmap(frame)
        if pixmap is not None:
            self.drop_zone.set_pixmap(pixmap)
        self._render_frame_faces(annotations)

    def _update_live_result_history(self, payload: dict[str, Any]) -> str | None:
        candidate: tuple[str, float] | None = None
        pipeline = str(payload.get("pipeline", "")).lower()
        if str(payload.get("decision", "")).lower() == "match":
            results = payload.get("results", [])
            if results and isinstance(results[0], dict):
                label_value = results[0].get("label")
                score_value = results[0].get("score")
                try:
                    score = float(score_value)
                    threshold = float(payload.get("threshold_used", 0.0))
                except (TypeError, ValueError):
                    score = 0.0
                    threshold = 0.0

                # Custom live search uses IVF-PQ over a very large synthetic
                # index, so low-confidence matches are allowed to draw a box
                # but must not be promoted to a visible identity.
                required_score = max(threshold, 0.30 if pipeline == "custom" else threshold)
                if label_value and score >= required_score:
                    candidate = (str(label_value), score)

        self._live_result_history.append(candidate)
        self._live_result_history = self._live_result_history[-5:]
        labels = [item[0] for item in self._live_result_history if item]
        for label in set(labels):
            scores = [
                item[1]
                for item in self._live_result_history
                if item and item[0] == label
            ]
            if pipeline == "custom":
                stable = (
                    (len(scores) >= 2 and max(scores) >= 0.34)
                    or (len(scores) >= 3 and max(scores) >= 0.26)
                )
            else:
                stable = len(scores) >= 2
            if stable:
                self._live_stable_label = label
                self._live_stable_misses = 0
                return label

        if self._live_stable_label:
            if candidate and candidate[0] == self._live_stable_label:
                self._live_stable_misses = 0
            else:
                self._live_stable_misses += 1
            if self._live_stable_misses < 4:
                return self._live_stable_label
            self._live_stable_label = None
            self._live_stable_misses = 0
        return None

    def _extract_annotations(
        self,
        payload: dict[str, Any],
        bbox_scale: tuple[float, float] = (1.0, 1.0),
    ) -> list[dict[str, Any]]:
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        thresholds: dict[int, float] = {}

        threshold = float(payload.get("threshold_used", 0.0))
        for result in payload.get("results", []):
            row = dict(result)
            row["face_bbox"] = scale_bbox(row.get("face_bbox"), bbox_scale)
            grouped[int(result.get("face_index", 0))].append(row)
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

        detected_faces = payload.get("detected_faces", [])

        for face_info in detected_faces:
            face_index = int(face_info.get("face_index", 0))
            if face_index in matched_face_indices:
                continue
            annotations.append(
                {
                    "face_index": face_index,
                    "face_bbox": scale_bbox(face_info.get("face_bbox"), bbox_scale),
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
