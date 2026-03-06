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
        self.setStyleSheet(app_stylesheet())

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Шапка
        header = QWidget()
        header.setStyleSheet("background-color: #27282c; border-bottom: 1px solid #3a3b40;")
        header.setFixedHeight(52)
        hbox = QHBoxLayout(header)
        hbox.setContentsMargins(20, 0, 20, 0)

        title = QLabel("Face Search")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #e1e2e6; background: transparent;")
        subtitle = QLabel("biometric identification system")
        subtitle.setStyleSheet("font-size: 12px; color: #8b8d93; background: transparent; margin-left: 8px;")

        hbox.addWidget(title)
        hbox.addWidget(subtitle)
        hbox.addStretch()

        # Индикатор подключения
        self._conn_dot = QLabel("●")
        self._conn_dot.setStyleSheet("color: #8b8d93; font-size: 16px; background: transparent;")
        self._conn_label = QLabel("checking...")
        self._conn_label.setStyleSheet("color: #8b8d93; font-size: 12px; background: transparent;")
        hbox.addWidget(self._conn_dot)
        hbox.addWidget(self._conn_label)

        root.addWidget(header)

        # Вкладки
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(SearchTab(), "  Search  ")
        tabs.addTab(EnrollTab(), "  Enroll  ")
        tabs.addTab(PersonsTab(), "  Persons  ")
        tabs.addTab(StatsTab(), "  Stats  ")
        root.addWidget(tabs, 1)

        self.setCentralWidget(central)
        self._check_backend()

    def _check_backend(self) -> None:
        url = self._settings.base_url.rstrip("/")
        try:
            client = ApiClient(self._settings)
            data = client.health()
            backend = data.get("embedding_backend", "?")
            self._conn_dot.setStyleSheet("color: #3ba55d; font-size: 16px; background: transparent;")
            self._conn_label.setText(f"{backend} · {url}")
            self._conn_label.setStyleSheet("color: #8b8d93; font-size: 12px; background: transparent;")
        except Exception:
            self._conn_dot.setStyleSheet("color: #ed4245; font-size: 16px; background: transparent;")
            self._conn_label.setText(f"offline · {url}")
            self._conn_label.setStyleSheet("color: #ed4245; font-size: 12px; background: transparent;")
