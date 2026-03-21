from __future__ import annotations

from functools import partial
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from app.core.api_client import ApiClient
from app.core.worker import ApiWorker
from app.ui.activity import recent_events
from app.ui.dialogs import show_error
from app.ui.widgets import ActionButton, Card, DimLabel, MetricCard, SectionHeading, StatusPill


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

        header_row = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(4)

        title = QLabel("Dashboard")
        title.setObjectName("brandTitle")
        title.setStyleSheet("font-size: 34px;")
        subtitle = DimLabel("Overview of backend health, indexes, and recent activity")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)

        header_row.addLayout(title_col)
        header_row.addStretch()

        self.status_pill = StatusPill("CHECKING LINK", state="idle")
        header_row.addWidget(self.status_pill)
        root.addLayout(header_row)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(14)
        metrics.setVerticalSpacing(14)
        self.metric_pipelines = MetricCard("Active pipelines", "-")
        self.metric_profiles = MetricCard("Total profiles", "-")
        self.metric_pretrained = MetricCard("Pretrained index", "-")
        self.metric_custom = MetricCard("Custom index", "-")
        metrics.addWidget(self.metric_pipelines, 0, 0)
        metrics.addWidget(self.metric_profiles, 0, 1)
        metrics.addWidget(self.metric_pretrained, 0, 2)
        metrics.addWidget(self.metric_custom, 0, 3)
        root.addLayout(metrics)

        main_row = QHBoxLayout()
        main_row.setSpacing(18)

        hero_card = Card()
        hero = hero_card.body()
        hero.addWidget(SectionHeading("Overview"))

        self.hero_headline = QLabel("Ready for search, enroll, and compare")
        self.hero_headline.setObjectName("metricValue")
        self.hero_headline.setStyleSheet("font-size: 28px; background: transparent;")
        self.hero_meta = DimLabel(
            "Use Face Search for live queries, enroll new profiles, or compare both models on the same image."
        )
        hero.addWidget(self.hero_headline)
        hero.addWidget(self.hero_meta)
        hero.addSpacing(10)

        quick_row = QHBoxLayout()
        quick_row.setSpacing(10)
        search_btn = ActionButton("Open face search", primary=True)
        db_btn = ActionButton("Open database")
        logs_btn = ActionButton("Open logs")
        search_btn.clicked.connect(partial(self._navigate, "search"))
        db_btn.clicked.connect(partial(self._navigate, "database"))
        logs_btn.clicked.connect(partial(self._navigate, "logs"))
        quick_row.addWidget(search_btn)
        quick_row.addWidget(db_btn)
        quick_row.addWidget(logs_btn)
        quick_row.addStretch()
        hero.addLayout(quick_row)

        hero.addSpacing(10)
        self.snapshot_view = QLabel(
            "No data loaded yet.\n\nRefresh the dashboard to inspect backend health, index statistics, and recent activity."
        )
        self.snapshot_view.setObjectName("dropZoneSubtitle")
        self.snapshot_view.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.snapshot_view.setMinimumHeight(260)
        self.snapshot_view.setStyleSheet(
            "padding: 18px; border: 1px solid #243241; background-color: #0c1219;"
        )
        hero.addWidget(self.snapshot_view, 1)
        main_row.addWidget(hero_card, 3)

        activity_card = Card()
        ac = activity_card.body()
        ac.addWidget(SectionHeading("Recent activity"))
        self.activity_layout = QVBoxLayout()
        self.activity_layout.setSpacing(10)
        ac.addLayout(self.activity_layout)
        ac.addStretch()
        main_row.addWidget(activity_card, 2)

        root.addLayout(main_row, 1)

    def _navigate(self, key: str) -> None:
        if self._navigate_to is not None:
            self._navigate_to(key)

    def _gather_dashboard(self) -> dict:
        health = self.api.health()
        persons = self.api.list_persons()
        pipelines = health.get("available_pipelines", [])
        stats = {}
        for pipeline in pipelines:
            stats[pipeline] = self.api.index_stats(pipeline)
        return {
            "health": health,
            "persons": persons,
            "stats": stats,
        }

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

        available = health.get("available_pipelines", [])
        self.status_pill.set_state("ok", f"ONLINE / {' + '.join(str(v).upper() for v in available)}")
        self.metric_pipelines.set_value(str(len(available)), f"Default: {health.get('default_pipeline', '-')}")
        self.metric_profiles.set_value(str(len(persons)), "Active records")

        pretrained = stats.get("pretrained", {})
        custom = stats.get("custom", {})
        self.metric_pretrained.set_value(
            str(pretrained.get("embeddings_count", 0)),
            f"{pretrained.get('model_name', 'onnx')} / {pretrained.get('index_type', '-')}",
        )
        custom_detail = f"{custom.get('model_name', 'n/a')} / {custom.get('index_type', '-')}"
        self.metric_custom.set_value(str(custom.get("embeddings_count", 0)), custom_detail)

        self.snapshot_view.setText(
            "\n".join(
                [
                    f"DEFAULT PIPELINE   {health.get('default_pipeline', '-')}",
                    f"MULTI-FACE SEARCH  {health.get('multi_face_search', False)}",
                    f"STRICT ENROLL      {health.get('strict_single_face_enroll', False)}",
                    "",
                    f"PRETRAINED LOADED  {pretrained.get('loaded', False)}",
                    f"PRETRAINED PATH    {pretrained.get('file_path', '-')}",
                    "",
                    f"CUSTOM LOADED      {custom.get('loaded', False)}",
                    f"CUSTOM PATH        {custom.get('file_path', '-')}",
                    "",
                    "RECENT PROFILES",
                    *[
                        f"  - {(person.get('label') or 'UNNAMED')} :: {person.get('id', '')[:12]}"
                        for person in persons[:6]
                    ],
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

        events = recent_events(limit=7)
        if not events:
            empty = DimLabel("No recent activity yet.")
            self.activity_layout.addWidget(empty)
            return

        for event in events:
            row = Card(variant="default")
            body = row.body()
            severity = event.severity
            body.addWidget(DimLabel(event.timestamp.strftime("%Y-%m-%d %H:%M:%S")))
            title = QLabel(f"{severity}  {event.category.upper()}")
            title.setObjectName("topbarTitle")
            body.addWidget(title)
            body.addWidget(DimLabel(event.message))
            if event.details:
                detail = DimLabel(event.details[:140])
                body.addWidget(detail)
            self.activity_layout.addWidget(row)

        self.activity_layout.addStretch()

    def _on_error(self, error: str) -> None:
        self.status_pill.set_state("error", "NODE OFFLINE")
        self.metric_pipelines.set_value("-", "Health request failed")
        show_error(self, "Dashboard refresh failed", error)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._built_once:
            self._built_once = True
        self._load()
