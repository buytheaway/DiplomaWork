"""Main application window with tabbed interface."""

from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QTabWidget

from app.ui.tabs.enroll_tab import EnrollTab
from app.ui.tabs.persons_tab import PersonsTab
from app.ui.tabs.search_tab import SearchTab
from app.ui.tabs.stats_tab import StatsTab


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Fast Biometric Face Search")
        self.resize(900, 600)

        tabs = QTabWidget()
        tabs.addTab(EnrollTab(), "Enroll")
        tabs.addTab(SearchTab(), "Search")
        tabs.addTab(PersonsTab(), "Persons")
        tabs.addTab(StatsTab(), "Stats")

        self.setCentralWidget(tabs)
