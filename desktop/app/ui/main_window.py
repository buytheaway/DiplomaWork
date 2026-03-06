from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.api_client import ApiClient
from app.core.config import DesktopSettings
from app.ui.tabs.enroll_tab import EnrollTab
from app.ui.tabs.persons_tab import PersonsTab
from app.ui.tabs.search_tab import SearchTab
from app.ui.tabs.stats_tab import StatsTab
from app.ui.theme import app_stylesheet


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._settings = DesktopSettings()
        self.setWindowTitle("Fast Biometric Face Search")
        self.resize(1020, 680)
        self.setMinimumSize(820, 520)

        # Применяем тему
        self.setStyleSheet(app_stylesheet())

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── header bar ────────────────────────────────────────────────
        header = QWidget()
        header.setObjectName("appHeader")
        header_lay = QHBoxLayout(header)
        header_lay.setContentsMargins(16, 8, 16, 8)

        title_col = QVBoxLayout()
        title_col.setSpacing(1)
        title = QLabel("Biometric Face Search")
        title.setObjectName("appTitle")
        subtitle = QLabel("Diploma project — vector-based face matching")
        subtitle.setObjectName("appSubtitle")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)

        self._conn_indicator = QLabel()
        self._conn_indicator.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        header_lay.addLayout(title_col)
        header_lay.addStretch()
        header_lay.addWidget(self._conn_indicator)

        root.addWidget(header)

        # ── tabs ──────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(EnrollTab(), "  Enroll  ")
        self.tabs.addTab(SearchTab(), "  Search  ")
        self.tabs.addTab(PersonsTab(), "  Persons  ")
        self.tabs.addTab(StatsTab(), "  Stats  ")

        root.addWidget(self.tabs, 1)

        self.setCentralWidget(central)

        # Проверяем бэкенд
        self._check_backend()

    def _check_backend(self) -> None:
        url = self._settings.base_url.rstrip("/")
        sb = self.statusBar()
        try:
            client = ApiClient(self._settings)
            data = client.health()
            backend = data.get("embedding_backend", "?")
            sb.showMessage(f"  {url}  \u00b7  backend: {backend}")
            self._conn_indicator.setText(f"\u25cf Connected  \u00b7  {backend}")
            self._conn_indicator.setStyleSheet("color: #34c759; font-size: 12px;")
        except Exception:
            sb.showMessage(f"  \u26a0 Backend unavailable: {url}")
            self._conn_indicator.setText("\u25cb Disconnected")
            self._conn_indicator.setStyleSheet("color: #ff453a; font-size: 12px;")
