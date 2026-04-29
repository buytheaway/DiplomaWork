from __future__ import annotations

from functools import partial

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMainWindow, QStackedWidget, QVBoxLayout, QWidget

from app.core.api_client import ApiClient
from app.core.config import DesktopSettings
from app.ui.activity import record_event
from app.ui.tabs.dashboard_tab import DashboardTab
from app.ui.tabs.persons_tab import PersonsTab
from app.ui.tabs.search_tab import SearchTab
from app.ui.tabs.stats_tab import StatsTab
from app.ui.theme import app_stylesheet
from app.ui.widgets import NavButton, StatusPill


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._settings = DesktopSettings()
        self._client = ApiClient(self._settings)
        self._pages: dict[str, QWidget] = {}
        self._nav_buttons: dict[str, NavButton] = {}
        self._last_health_ok: bool | None = None

        self.setWindowTitle("Fast Biometric Face Search")
        self.resize(1360, 880)
        self.setMinimumSize(1180, 760)
        self.setStyleSheet(app_stylesheet())

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_sidebar())

        content_shell = QWidget()
        content_layout = QVBoxLayout(content_shell)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(self._build_topbar())

        self.stack = QStackedWidget()
        self._pages = {
            "dashboard": DashboardTab(self._navigate_to),
            "search": SearchTab(),
            "database": PersonsTab(),
            "logs": StatsTab(),
        }
        for page in self._pages.values():
            self.stack.addWidget(page)
        content_layout.addWidget(self.stack, 1)

        root.addWidget(content_shell, 1)
        self.setCentralWidget(central)

        self._navigate_to("dashboard")
        self._check_backend()

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._check_backend)
        self._status_timer.start(15000)

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(208)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 18, 16, 18)
        layout.setSpacing(12)

        brand = QLabel("Fast Face Search")
        brand.setObjectName("brandTitle")
        subtitle = QLabel("Operator desktop")
        subtitle.setObjectName("brandSubtitle")
        layout.addWidget(brand)
        layout.addWidget(subtitle)
        layout.addSpacing(10)

        for key, title in [
            ("dashboard", "Dashboard"),
            ("search", "Face Search"),
            ("database", "Database"),
            ("logs", "Logs"),
        ]:
            button = NavButton(title)
            button.clicked.connect(partial(self._navigate_to, key))
            self._nav_buttons[key] = button
            layout.addWidget(button)

        layout.addStretch()

        self._sidebar_meta = QLabel(self._settings.base_url.rstrip("/"))
        self._sidebar_meta.setObjectName("operatorMeta")
        layout.addWidget(self._sidebar_meta)

        return sidebar

    def _build_topbar(self) -> QWidget:
        topbar = QFrame()
        topbar.setObjectName("topbar")
        topbar.setFixedHeight(64)

        layout = QHBoxLayout(topbar)
        layout.setContentsMargins(22, 10, 22, 10)
        layout.setSpacing(12)

        self._topbar_title = QLabel("Dashboard")
        self._topbar_title.setObjectName("topbarTitle")
        layout.addWidget(self._topbar_title)
        layout.addStretch()

        self._header_status = StatusPill("Offline", state="idle")
        layout.addWidget(self._header_status)
        return topbar

    def _navigate_to(self, key: str) -> None:
        page = self._pages[key]
        self.stack.setCurrentWidget(page)
        for name, button in self._nav_buttons.items():
            button.setChecked(name == key)

        titles = {
            "dashboard": "Dashboard",
            "search": "Face search",
            "database": "Database",
            "logs": "Logs",
        }
        self._topbar_title.setText(titles.get(key, key.replace("_", " ").title()))
        record_event("ui", f"Switched page to {key}", severity="INFO")

    def _check_backend(self) -> None:
        url = self._settings.base_url.rstrip("/")
        try:
            data = self._client.health()
            backend = str(data.get("embedding_backend", "?"))
            pipelines = [str(item) for item in data.get("available_pipelines", [])]
            pipeline_text = ", ".join(pipelines) if pipelines else "no pipelines"
            self._header_status.set_state("ok", f"{backend} - {pipeline_text}")
            if self._last_health_ok is not True:
                record_event("ui", "Backend link established", severity="INFO", details=url)
            self._last_health_ok = True
        except Exception as exc:  # noqa: BLE001
            self._header_status.set_state("error", "Backend offline")
            if self._last_health_ok is not False:
                record_event("ui", "Backend health check failed", severity="ERROR", details=str(exc))
            self._last_health_ok = False
