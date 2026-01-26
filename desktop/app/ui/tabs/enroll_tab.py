import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.api_client import ApiClient, format_api_error


class EnrollTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.api = ApiClient()
        self.image_path = QLineEdit()
        self.label_input = QLineEdit()
        self.response_view = QTextEdit()
        self.response_view.setReadOnly(True)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse)

        enroll_btn = QPushButton("Enroll")
        enroll_btn.clicked.connect(self._enroll)

        file_row = QHBoxLayout()
        file_row.addWidget(self.image_path)
        file_row.addWidget(browse_btn)

        form = QFormLayout()
        form.addRow(QLabel("Image"), file_row)
        form.addRow("Label", self.label_input)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(enroll_btn, alignment=Qt.AlignLeft)
        layout.addWidget(QLabel("Response"))
        layout.addWidget(self.response_view)
        self.setLayout(layout)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select image")
        if path:
            self.image_path.setText(path)

    def _enroll(self) -> None:
        path = self.image_path.text().strip()
        if not path:
            QMessageBox.warning(self, "Missing", "Select an image file")
            return
        label = self.label_input.text().strip() or None
        try:
            response = self.api.enroll(path, label)
            pretty = json.dumps(response, indent=2)
            self.response_view.setPlainText(pretty)
        except Exception as exc:
            QMessageBox.critical(self, "Enroll failed", format_api_error(exc))
