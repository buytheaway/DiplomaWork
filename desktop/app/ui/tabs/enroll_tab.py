# Вкладка регистрации лица
from __future__ import annotations

import json

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.api_client import ApiClient
from app.core.worker import ApiWorker
from app.ui.dialogs import show_error, show_warning
from app.ui.widgets import Card, DimLabel, ImagePreview, SectionHeading


class EnrollTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.api = ApiClient()
        self._worker: ApiWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        top_row = QHBoxLayout()
        top_row.setSpacing(16)

        form_card = Card()
        fc = form_card.body()
        fc.addWidget(SectionHeading("Register new face"))
        fc.addSpacing(4)

        fc.addWidget(DimLabel("Image file"))
        file_row = QHBoxLayout()
        file_row.setSpacing(8)
        self.image_path = QLineEdit()
        self.image_path.setPlaceholderText("Select an image...")
        self.image_path.setReadOnly(True)
        file_row.addWidget(self.image_path, 1)

        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse)
        file_row.addWidget(browse_btn)
        fc.addLayout(file_row)

        fc.addSpacing(8)

        fc.addWidget(DimLabel("Label (optional — имя или ID человека)"))
        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("e.g. Alice, employee_042")
        fc.addWidget(self.label_input)

        fc.addSpacing(8)

        fc.addWidget(DimLabel("Pipeline: both pretrained and custom"))

        fc.addSpacing(12)

        self.enroll_btn = QPushButton("Enroll")
        self.enroll_btn.setObjectName("primary")
        self.enroll_btn.setFixedHeight(36)
        self.enroll_btn.clicked.connect(self._enroll)
        fc.addWidget(self.enroll_btn)

        self.status_label = DimLabel("")
        fc.addWidget(self.status_label)
        fc.addStretch()

        top_row.addWidget(form_card, 3)

        preview_card = Card()
        pc = preview_card.body()
        pc.addWidget(SectionHeading("Preview"))
        self.preview = ImagePreview(180)
        pc.addWidget(self.preview)
        pc.addStretch()
        top_row.addWidget(preview_card, 1)

        root.addLayout(top_row, 1)

        result_card = Card()
        rc = result_card.body()
        rc.addWidget(SectionHeading("Response"))
        self.response_view = QTextEdit()
        self.response_view.setReadOnly(True)
        self.response_view.setMaximumHeight(180)
        self.response_view.setPlaceholderText("Enrollment result will appear here...")
        rc.addWidget(self.response_view)
        root.addWidget(result_card)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select image", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All files (*.*)",
        )
        if path:
            self.image_path.setText(path)
            self.preview.load(path)

    def _enroll(self) -> None:
        path = self.image_path.text().strip()
        if not path:
            show_warning(self, "Missing", "Select an image file")
            return
        label = self.label_input.text().strip() or None
        self.enroll_btn.setEnabled(False)
        self.status_label.setText("Enrolling...")
        self._worker = ApiWorker(self.api.enroll, path, label, "both", parent=self)
        self._worker.finished.connect(self._on_success)
        self._worker.failed.connect(self._on_error)
        self._worker.start()

    def _on_success(self, result: object) -> None:
        self.enroll_btn.setEnabled(True)
        self.status_label.setText("")
        self.response_view.setPlainText(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    def _on_error(self, error: str) -> None:
        self.enroll_btn.setEnabled(True)
        self.status_label.setText("")
        show_error(self, "Enroll failed", error)
