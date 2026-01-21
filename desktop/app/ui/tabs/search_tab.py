from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.api_client import ApiClient


class SearchTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.api = ApiClient()
        self.image_path = QLineEdit()
        self.k_input = QSpinBox()
        self.k_input.setRange(1, 100)
        self.k_input.setValue(5)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse)

        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self._search)

        file_row = QHBoxLayout()
        file_row.addWidget(self.image_path)
        file_row.addWidget(browse_btn)

        form = QFormLayout()
        form.addRow(QLabel("Image"), file_row)
        form.addRow("Top K", self.k_input)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Person ID", "Label", "Score", "Distance"])

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(search_btn, alignment=Qt.AlignLeft)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select image")
        if path:
            self.image_path.setText(path)

    def _search(self) -> None:
        path = self.image_path.text().strip()
        if not path:
            QMessageBox.warning(self, "Missing", "Select an image file")
            return
        k = self.k_input.value()
        try:
            response = self.api.search(path, k)
            results = response.get("results", [])
            self.table.setRowCount(len(results))
            for row_idx, result in enumerate(results):
                self.table.setItem(row_idx, 0, QTableWidgetItem(result.get("person_id", "")))
                self.table.setItem(row_idx, 1, QTableWidgetItem(result.get("label") or ""))
                self.table.setItem(row_idx, 2, QTableWidgetItem(str(result.get("score"))))
                self.table.setItem(row_idx, 3, QTableWidgetItem(str(result.get("distance"))))
        except Exception as exc:
            QMessageBox.critical(self, "Search failed", str(exc))
