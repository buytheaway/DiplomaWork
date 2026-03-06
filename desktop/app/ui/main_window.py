from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QTabWidget

from app.core.api_client import ApiClient
from app.core.config import DesktopSettings
from app.ui.tabs.enroll_tab import EnrollTab
from app.ui.tabs.persons_tab import PersonsTab
from app.ui.tabs.search_tab import SearchTab
from app.ui.tabs.stats_tab import StatsTab


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._settings = DesktopSettings()
        self.setWindowTitle("Fast Biometric Face Search")
        self.resize(900, 600)

        tabs = QTabWidget()
        tabs.addTab(EnrollTab(), "Enroll")
        tabs.addTab(SearchTab(), "Search")
        tabs.addTab(PersonsTab(), "Persons")
        tabs.addTab(StatsTab(), "Stats")
        self.setCentralWidget(tabs)

        # Показываем URL бэкенда и статус подключения
        self._check_backend()

    def _check_backend(self) -> None:
        url = self._settings.base_url.rstrip("/")
        sb = self.statusBar()
        try:
            client = ApiClient(self._settings)
            data = client.health()
            backend = data.get("embedding_backend", "?")
            sb.showMessage(f"Connected: {url}  |  backend: {backend}")
        except Exception:
            sb.showMessage(f"⚠ Backend unavailable: {url}")
