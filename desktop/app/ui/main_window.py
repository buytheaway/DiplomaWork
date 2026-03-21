from __future__ import annotations

from functools import partial

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.api_client import ApiClient
from app.core.config import DesktopSettings
from app.ui.activity import record_event
from app.ui.tabs.dashboard_tab import DashboardTab
from app.ui.tabs.persons_tab import PersonsTab
from app.ui.tabs.search_tab import SearchTab
from app.ui.tabs.stats_tab import StatsTab
from app.ui.theme import app_stylesheet
from app.ui.widgets import ActionButton, NavButton, StatusPill


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._settings = DesktopSettings()
        self._client = ApiClient(self._settings)
        self._pages: dict[str, QWidget] = {}
        self._nav_buttons: dict[str, NavButton] = {}
        self._last_health_ok: bool | None = None

        self.setWindowTitle("Fast Biometric Face Search")
        self.resize(1460, 900)
        self.setMinimumSize(1220, 760)
        self.setStyleSheet(app_stylesheet())

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sidebar = self._build_sidebar()
        root.addWidget(sidebar)

        content_shell = QWidget()
        content_layout = QVBoxLayout(content_shell)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        topbar = self._build_topbar()
        content_layout.addWidget(topbar)

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
        sidebar.setFixedWidth(230)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        brand = QLabel("FAST FACE SEARCH")
        brand.setObjectName("brandTitle")
        subtitle = QLabel("Desktop client")
        subtitle.setObjectName("brandSubtitle")
        layout.addWidget(brand)
        layout.addWidget(subtitle)
        layout.addSpacing(12)

        operator_card = QFrame()
        operator_card.setObjectName("card")
        operator_card.setProperty("variant", "default")
        operator_layout = QVBoxLayout(operator_card)
        operator_layout.setContentsMargins(14, 14, 14, 14)
        operator_layout.setSpacing(4)
        operator_name = QLabel("Connected endpoint")
        operator_name.setObjectName("operatorName")
        operator_meta = QLabel(self._settings.base_url.rstrip("/"))
        operator_meta.setObjectName("operatorMeta")
        operator_layout.addWidget(operator_name)
        operator_layout.addWidget(operator_meta)
        layout.addWidget(operator_card)
        layout.addSpacing(6)

        group = QButtonGroup(self)
        for key, title in [
            ("dashboard", "Dashboard"),
            ("search", "Face search"),
            ("database", "Database"),
            ("logs", "Logs"),
        ]:
            button = NavButton(title)
            button.clicked.connect(partial(self._navigate_to, key))
            group.addButton(button)
            self._nav_buttons[key] = button
            layout.addWidget(button)

        layout.addStretch()

        scan_btn = ActionButton("Open search", primary=True)
        scan_btn.clicked.connect(partial(self._navigate_to, "search"))
        layout.addWidget(scan_btn)

        self._sidebar_health = StatusPill("LINK UNKNOWN", state="idle")
        layout.addWidget(self._sidebar_health)

        logout_btn = QPushButton("Close session")
        logout_btn.setObjectName("secondaryButton")
        logout_btn.setEnabled(False)
        layout.addWidget(logout_btn)

        return sidebar

    def _build_topbar(self) -> QWidget:
        topbar = QFrame()
        topbar.setObjectName("topbar")
        topbar.setFixedHeight(68)

        layout = QHBoxLayout(topbar)
        layout.setContentsMargins(22, 12, 22, 12)
        layout.setSpacing(12)

        self.global_search = QLineEdit()
        self.global_search.setPlaceholderText("Filter current page")
        self.global_search.textChanged.connect(self._apply_global_filter)
        layout.addWidget(self.global_search, 1)

        self._topbar_title = QLabel("DASHBOARD")
        self._topbar_title.setObjectName("topbarTitle")
        layout.addWidget(self._topbar_title)

        self._header_status = StatusPill("OFFLINE", state="idle")
        layout.addWidget(self._header_status)

        return topbar

    def _navigate_to(self, key: str) -> None:
        page = self._pages[key]
        self.stack.setCurrentWidget(page)
        for name, button in self._nav_buttons.items():
            button.setChecked(name == key)
        self._topbar_title.setText(key.upper().replace("_", " "))
        placeholders = {
            "dashboard": "Filter overview",
            "search": "Filter results",
            "database": "Filter by label or person id",
            "logs": "Filter events",
        }
        self.global_search.setPlaceholderText(placeholders.get(key, "SEARCH"))
        self._apply_global_filter(self.global_search.text())
        record_event("ui", f"Switched page to {key}", severity="INFO")

    def _apply_global_filter(self, text: str) -> None:
        current = self.stack.currentWidget()
        if hasattr(current, "apply_global_filter"):
            current.apply_global_filter(text)

    def _check_backend(self) -> None:
        url = self._settings.base_url.rstrip("/")
        try:
            data = self._client.health()
            backend = data.get("embedding_backend", "?")
            pipelines = data.get("available_pipelines", [])
            joined = " + ".join(str(item).upper() for item in pipelines) if pipelines else "NO PIPELINES"
            self._header_status.set_state("ok", f"{backend.upper()} / {joined}")
            self._sidebar_health.set_state("ok", f"ONLINE  {url}")
            if self._last_health_ok is not True:
                record_event("ui", "Backend link established", severity="INFO", details=url)
            self._last_health_ok = True
        except Exception as exc:  # noqa: BLE001
            self._header_status.set_state("error", "BACKEND OFFLINE")
            self._sidebar_health.set_state("error", f"OFFLINE  {url}")
            if self._last_health_ok is not False:
                record_event("ui", "Backend health check failed", severity="ERROR", details=str(exc))
            self._last_health_ok = False
