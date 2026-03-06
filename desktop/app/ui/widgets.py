# Переиспользуемые UI-блоки — карточка, info-row, badge, image preview.
# Минимум абстракций, максимум пользы.

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class Card(QFrame):
    # Визуальная группировка — по сути просто QFrame с классом .card
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("class", "card")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(16, 14, 16, 14)
        self._layout.setSpacing(10)

    def body(self) -> QVBoxLayout:
        return self._layout


class SectionHeading(QLabel):
    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setProperty("class", "sectionHeading")


class DimLabel(QLabel):
    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setProperty("class", "dimLabel")


class InfoRow(QWidget):
    # Строка «ключ — значение» для отображения статистики
    def __init__(self, key: str, value: str = "-", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)

        self._key = DimLabel(key)
        self._value = QLabel(value)
        self._value.setStyleSheet("font-weight: 500;")

        lay.addWidget(self._key)
        lay.addStretch()
        lay.addWidget(self._value)

    def set_value(self, text: str) -> None:
        self._value.setText(text)


class StatBox(QWidget):
    # Крупное число + мелкая подпись (для dashboard)
    def __init__(self, key: str, value: str = "-", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(2)
        lay.setAlignment(Qt.AlignCenter)

        self._value = QLabel(value)
        self._value.setProperty("class", "infoValue")
        self._value.setAlignment(Qt.AlignCenter)

        self._key = QLabel(key)
        self._key.setProperty("class", "infoKey")
        self._key.setAlignment(Qt.AlignCenter)

        lay.addWidget(self._value)
        lay.addWidget(self._key)

    def set_value(self, text: str) -> None:
        self._value.setText(text)


class ImagePreview(QLabel):
    # Превью выбранного изображения (фиксированный размер)
    MAX_W = 160
    MAX_H = 160

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("imagePreview")
        self.setFixedSize(self.MAX_W, self.MAX_H)
        self.setAlignment(Qt.AlignCenter)
        self._show_placeholder()

    def load(self, path: str) -> None:
        p = Path(path)
        if not p.is_file():
            self._show_placeholder()
            return
        pix = QPixmap(str(p))
        if pix.isNull():
            self._show_placeholder()
            return
        scaled = pix.scaled(
            self.MAX_W - 4, self.MAX_H - 4,
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def clear_preview(self) -> None:
        self._show_placeholder()

    def _show_placeholder(self) -> None:
        self.clear()
        self.setText("No image")
        self.setStyleSheet(
            "color: #5a5b60; font-size: 11px; font-style: italic;"
        )
