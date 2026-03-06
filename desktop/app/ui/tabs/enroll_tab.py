from __future__ import annotations

import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.api_client import ApiClient
from app.core.worker import ApiWorker
from app.ui.widgets import Card, DimLabel, ImagePreview, SectionHeading


class EnrollTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.api = ApiClient()
        self._worker: ApiWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        # ── верхняя часть: выбор файла + превью ───────────────────────
        top_row = QHBoxLayout()
        top_row.setSpacing(16)

        # Левая колонка — форма
        form_card = Card()
        form = form_card.body()

        form.addWidget(SectionHeading("Enroll a face"))
        form.addWidget(DimLabel("Select an image and optionally assign a label"))
        form.addSpacing(6)

        # image path
        path_label = DimLabel("Image file")
        form.addWidget(path_label)

        path_row = QHBoxLayout()
        path_row.setSpacing(8)
        self.image_path = QLineEdit()
        self.image_path.setPlaceholderText("Choose an image…")
        self.image_path.setReadOnly(True)
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse)
        path_row.addWidget(self.image_path, 1)
        path_row.addWidget(browse_btn)
        form.addLayout(path_row)

        form.addSpacing(4)

        # label
        label_label = DimLabel("Label (optional)")
        form.addWidget(label_label)
        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("e.g. Alice, Bob")
        form.addWidget(self.label_input)

        form.addSpacing(8)

        # action button
        self.enroll_btn = QPushButton("Enroll")
        self.enroll_btn.setObjectName("primary")
        self.enroll_btn.setFixedWidth(140)
        self.enroll_btn.clicked.connect(self._enroll)
        form.addWidget(self.enroll_btn)

        self.status_label = DimLabel("")
        form.addWidget(self.status_label)

        form.addStretch()

        # Правая колонка — превью
        preview_card = Card()
        pv = preview_card.body()
        pv.setAlignment(Qt.AlignCenter)
        pv.addWidget(DimLabel("Preview"))
        self.preview = ImagePreview()
        pv.addWidget(self.preview, alignment=Qt.AlignCenter)
        pv.addStretch()

        top_row.addWidget(form_card, 3)
        top_row.addWidget(preview_card, 1)

        root.addLayout(top_row)

        # ── результат ────────────────────────────────────────────────
        result_card = Card()
        rc = result_card.body()
        rc.addWidget(SectionHeading("Response"))
        self.response_view = QTextEdit()
        self.response_view.setReadOnly(True)
        self.response_view.setMaximumHeight(160)
        self.response_view.setPlaceholderText("Enrollment result will appear here…")
        rc.addWidget(self.response_view)

        root.addWidget(result_card)
        root.addStretch()

    # ── actions ──────────────────────────────────────────────────────

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select image", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All files (*)",
        )
        if path:
            self.image_path.setText(path)
            self.preview.load(path)

    def _enroll(self) -> None:
        path = self.image_path.text().strip()
        if not path:
            QMessageBox.warning(self, "Missing", "Select an image file")
            return
        label = self.label_input.text().strip() or None
        self.enroll_btn.setEnabled(False)
        self.status_label.setText("Enrolling\u2026")
        self._worker = ApiWorker(self.api.enroll, path, label, parent=self)
        self._worker.finished.connect(self._on_success)
        self._worker.failed.connect(self._on_error)
        self._worker.start()

    def _on_success(self, result: object) -> None:
        self.enroll_btn.setEnabled(True)
        self.status_label.setText("")
        self.response_view.setPlainText(json.dumps(result, indent=2, default=str))

    def _on_error(self, error: str) -> None:
        self.enroll_btn.setEnabled(True)
        self.status_label.setText("")
        QMessageBox.critical(self, "Enroll failed", error)
