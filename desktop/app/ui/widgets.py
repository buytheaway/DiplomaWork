# Переиспользуемые виджеты для desktop UI
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)


class Card(QFrame):
    # Карточка с тёмным фоном и скруглёнными углами
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setProperty("class", "card")  # для QSS
        self.setObjectName("card")
        self.setFrameShape(QFrame.NoFrame)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(16, 14, 16, 14)
        self._layout.setSpacing(8)

    def body(self) -> QVBoxLayout:
        return self._layout


class SectionHeading(QLabel):
    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self.setStyleSheet("font-size: 14px; font-weight: 600; color: #e1e2e6; background: transparent;")


class DimLabel(QLabel):
    # Бледный вспомогательный текст
    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(text, parent)
        self.setStyleSheet("color: #8b8d93; font-size: 12px; background: transparent;")


class InfoRow(QLabel):
    # Ключ: значение в одну строку
    def __init__(self, key: str, value: str = "-", parent=None) -> None:
        super().__init__(parent)
        self._key = key
        self.set_value(value)
        self.setStyleSheet("font-size: 13px; background: transparent;")

    def set_value(self, value: str) -> None:
        self.setText(f'<span style="color:#8b8d93">{self._key}:</span>  {value}')
        self.setTextFormat(Qt.RichText)


class StatBox(QLabel):
    # Большое число + подпись
    def __init__(self, label: str, value: str = "-", parent=None) -> None:
        super().__init__(parent)
        self._label = label
        self._value_text = value
        self._render()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: transparent;")

    def _render(self) -> None:
        self.setText(
            f'<div style="text-align:center">'
            f'<span style="font-size:22px; font-weight:700; color:#e1e2e6">{self._value_text}</span><br>'
            f'<span style="font-size:11px; color:#8b8d93">{self._label}</span>'
            f'</div>'
        )
        self.setTextFormat(Qt.RichText)

    def set_value(self, value: str) -> None:
        self._value_text = value
        self._render()


class ImagePreview(QLabel):
    # Превью картинки с фиксированным размером
    def __init__(self, size: int = 160, parent=None) -> None:
        super().__init__(parent)
        self._size = size
        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            f"background-color: #2c2d31; border: 1px solid #3a3b40;"
            f" border-radius: 6px; color: #8b8d93; font-size: 11px;"
        )
        self.setText("No image")

    def load(self, path: str) -> None:
        pix = QPixmap(path)
        if pix.isNull():
            self.setText("Invalid image")
            return
        self.setPixmap(
            pix.scaled(self._size, self._size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def clear_preview(self) -> None:
        self.clear()
        self.setText("No image")
