from __future__ import annotations

from functools import partial
from typing import Callable

from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.core.api_client import ApiClient
from app.core.worker import ApiWorker
from app.ui.activity import recent_events
from app.ui.dialogs import show_error
from app.ui.widgets import ActionButton, Card, DimLabel, MetricCard, SectionHeading


class DashboardTab(QWidget):
    def __init__(self, navigate_to: Callable[[str], None] | None = None) -> None:
        super().__init__()
        self.api = ApiClient()
        self._worker: ApiWorker | None = None
        self._navigate_to = navigate_to
        self._built_once = False
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(18)

        header = QVBoxLayout()
        header.setSpacing(4)
        title = QLabel("Dashboard")
        title.setObjectName("brandTitle")
        subtitle = DimLabel("Quick system summary and recent operator activity.")
        header.addWidget(title)
        header.addWidget(subtitle)
        root.addLayout(header)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(14)
        metrics.setVerticalSpacing(14)
        self.metric_pipelines = MetricCard("Pipelines", "-")
        self.metric_profiles = MetricCard("Profiles", "-")
        self.metric_vectors = MetricCard("Indexed vectors", "-")
        metrics.addWidget(self.metric_pipelines, 0, 0)
        metrics.addWidget(self.metric_profiles, 0, 1)
        metrics.addWidget(self.metric_vectors, 0, 2)
        root.addLayout(metrics)

        main_row = QHBoxLayout()
        main_row.setSpacing(18)

        overview_card = Card()
        overview = overview_card.body()
        overview.addWidget(SectionHeading("System overview"))

        self.overview_headline = QLabel("Waiting for backend response")
        self.overview_headline.setObjectName("metricValue")
        self.overview_headline.setStyleSheet("font-size: 26px; background: transparent;")
        self.overview_text = DimLabel(
            "Open Face Search to run queries, compare pipelines, or enroll new records."
        )
        overview.addWidget(self.overview_headline)
        overview.addWidget(self.overview_text)

        self.overview_lines = QLabel("No data loaded yet.")
        self.overview_lines.setObjectName("dropZoneSubtitle")
        self.overview_lines.setStyleSheet(
            "padding: 14px; border: 1px solid #25313d; border-radius: 8px; background-color: #0f151d;"
        )
        overview.addWidget(self.overview_lines)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        search_btn = ActionButton("Open face search", primary=True)
        database_btn = ActionButton("Open database")
        logs_btn = ActionButton("Open logs")
        search_btn.clicked.connect(partial(self._navigate, "search"))
        database_btn.clicked.connect(partial(self._navigate, "database"))
        logs_btn.clicked.connect(partial(self._navigate, "logs"))
        actions.addWidget(search_btn)
        actions.addWidget(database_btn)
        actions.addWidget(logs_btn)
        actions.addStretch()
        overview.addLayout(actions)
        main_row.addWidget(overview_card, 3)

        activity_card = Card()
        activity = activity_card.body()
        activity.addWidget(SectionHeading("Recent activity"))
        self.activity_layout = QVBoxLayout()
        self.activity_layout.setSpacing(10)
        activity.addLayout(self.activity_layout)
        activity.addStretch()
        main_row.addWidget(activity_card, 2)

        root.addLayout(main_row, 1)

    def _navigate(self, key: str) -> None:
        if self._navigate_to is not None:
            self._navigate_to(key)

    def _gather_dashboard(self) -> dict:
        health = self.api.health()
        persons = self.api.list_persons()
        stats = {}
        for pipeline in health.get("available_pipelines", []):
            stats[pipeline] = self.api.index_stats(pipeline)
        return {"health": health, "persons": persons, "stats": stats}

    def _load(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._worker = ApiWorker(self._gather_dashboard, parent=self)
        self._worker.finished.connect(self._on_success)
        self._worker.failed.connect(self._on_error)
        self._worker.start()

    def _on_success(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        health = payload.get("health", {})
        persons = payload.get("persons", [])
        stats = payload.get("stats", {})

        pipelines = [str(item) for item in health.get("available_pipelines", [])]
        total_vectors = sum(int(stats.get(name, {}).get("embeddings_count", 0)) for name in pipelines)
        self.metric_pipelines.set_value(str(len(pipelines)), ", ".join(pipelines) if pipelines else "Unavailable")
        self.metric_profiles.set_value(str(len(persons)), "Active records")
        self.metric_vectors.set_value(str(total_vectors), f"Default: {health.get('default_pipeline', '-')}")

        pretrained = stats.get("pretrained", {})
        custom = stats.get("custom", {})
        self.overview_headline.setText("System is ready" if pipelines else "Backend is reachable")
        self.overview_text.setText(
            "Use Face Search for the main workflow. Database and Logs remain secondary operational tools."
        )
        self.overview_lines.setText(
            "\n".join(
                [
                    f"Default pipeline: {health.get('default_pipeline', '-')}",
                    f"Multi-face search: {health.get('multi_face_search', False)}",
                    f"Single-face enroll: {health.get('strict_single_face_enroll', False)}",
                    f"Pretrained index: {pretrained.get('embeddings_count', 0)} vectors",
                    f"Custom index: {custom.get('embeddings_count', 0)} vectors",
                    f"Latest profile: {(persons[0].get('label') if persons else 'No records yet')}",
                ]
            )
        )
        self._render_activity()

    def _render_activity(self) -> None:
        while self.activity_layout.count():
            item = self.activity_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        events = recent_events(limit=6)
        if not events:
            self.activity_layout.addWidget(DimLabel("No recent activity yet."))
            return

        for event in events:
            row = Card(variant="subtle")
            body = row.body()
            body.setContentsMargins(12, 12, 12, 12)
            body.addWidget(DimLabel(event.timestamp.strftime("%Y-%m-%d %H:%M:%S")))
            title = QLabel(f"{event.category.title()} · {event.severity}")
            title.setObjectName("topbarTitle")
            title.setStyleSheet("font-size: 13px;")
            body.addWidget(title)
            body.addWidget(DimLabel(event.message))
            if event.details:
                body.addWidget(DimLabel(event.details[:140]))
            self.activity_layout.addWidget(row)

        self.activity_layout.addStretch()

    def _on_error(self, error: str) -> None:
        self.metric_pipelines.set_value("-", "Request failed")
        self.metric_profiles.set_value("-", "")
        self.metric_vectors.set_value("-", "")
        show_error(self, "Dashboard refresh failed", error)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._built_once:
            self._built_once = True
        self._load()
