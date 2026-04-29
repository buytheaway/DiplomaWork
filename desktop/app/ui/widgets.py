from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class Card(QFrame):
    def __init__(self, parent=None, *, variant: str = "default") -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.setProperty("variant", variant)
        self.setFrameShape(QFrame.NoFrame)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(18, 16, 18, 16)
        self._layout.setSpacing(10)

    def body(self) -> QVBoxLayout:
        return self._layout


class CollapsibleSection(QFrame):
    def __init__(self, title: str, parent=None, *, expanded: bool = False) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.setProperty("variant", "subtle")

        self._title = title
        self._expanded = expanded

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        self._toggle = QPushButton()
        self._toggle.setObjectName("collapseButton")
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.clicked.connect(self.set_expanded)
        layout.addWidget(self._toggle)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(10)
        layout.addWidget(self._content)

        self.set_expanded(expanded)

    def body(self) -> QVBoxLayout:
        return self._content_layout

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self._toggle.setChecked(expanded)
        arrow = "v" if expanded else ">"
        self._toggle.setText(f"{arrow}  {self._title}")
        self._content.setVisible(expanded)

    def is_expanded(self) -> bool:
        return self._expanded


class SectionHeading(QLabel):
    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self.setObjectName("sectionHeading")


class DimLabel(QLabel):
    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(text, parent)
        self.setObjectName("dimLabel")
        self.setWordWrap(True)


class InfoRow(QWidget):
    def __init__(self, label: str, value: str = "-", parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._label = QLabel(label)
        self._label.setObjectName("microLabel")
        self._value = QLabel(value)
        self._value.setObjectName("infoValue")
        self._value.setWordWrap(True)

        layout.addWidget(self._label)
        layout.addWidget(self._value)

    def set_value(self, value: str) -> None:
        self._value.setText(value)


class MetricCard(Card):
    def __init__(self, title: str, value: str = "-", detail: str = "", parent=None) -> None:
        super().__init__(parent, variant="metric")
        body = self.body()

        self.title_label = QLabel(title)
        self.title_label.setObjectName("metricTitle")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricValue")
        self.detail_label = QLabel(detail)
        self.detail_label.setObjectName("metricDetail")
        self.detail_label.setWordWrap(True)

        body.addWidget(self.title_label)
        body.addStretch()
        body.addWidget(self.value_label)
        body.addWidget(self.detail_label)

    def set_value(self, value: str, detail: str | None = None) -> None:
        self.value_label.setText(value)
        if detail is not None:
            self.detail_label.setText(detail)


class StatusPill(QLabel):
    def __init__(self, text: str = "Offline", *, state: str = "idle", parent=None) -> None:
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.set_state(state, text)

    def set_state(self, state: str, text: str | None = None) -> None:
        if text is not None:
            self.setText(text)
        self.setProperty("state", state)
        self.style().unpolish(self)
        self.style().polish(self)


class NavButton(QPushButton):
    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self.setObjectName("navButton")
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)


class ActionButton(QPushButton):
    def __init__(self, text: str, parent=None, *, primary: bool = False) -> None:
        super().__init__(text, parent)
        self.setObjectName("primaryButton" if primary else "secondaryButton")
        self.setCursor(Qt.PointingHandCursor)


class ConsoleView(QTextEdit):
    def __init__(self, placeholder: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setPlaceholderText(placeholder)
        self.setObjectName("consoleView")


class ImageDropZone(QFrame):
    fileDropped = Signal(str)

    def __init__(self, title: str = "Choose an image", parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("imageDropZone")
        self._pixmap: QPixmap | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignCenter)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("dropZoneTitle")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.subtitle_label = QLabel(
            "Drag a JPG, PNG, BMP or WEBP image here,\n"
            "or open it from local storage."
        )
        self.subtitle_label.setObjectName("dropZoneSubtitle")
        self.subtitle_label.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel("No image selected")
        self.image_label.setObjectName("dropZoneImage")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(260)

        layout.addStretch()
        layout.addWidget(self.image_label)
        layout.addWidget(self.title_label)
        layout.addWidget(self.subtitle_label)
        layout.addStretch()

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        mime = event.mimeData()
        if mime.hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if not urls:
            return
        local = urls[0].toLocalFile()
        if local:
            self.fileDropped.emit(local)
            event.acceptProposedAction()

    def load(self, path: str) -> None:
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.clear_preview()
            self.image_label.setText("Invalid image")
            return
        self.set_pixmap(pixmap)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        scaled = pixmap.scaled(460, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def clear_preview(self) -> None:
        self._pixmap = None
        self.image_label.clear()
        self.image_label.setText("No image selected")


class ResultCard(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("resultCard")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self.thumb = QLabel("Face")
        self.thumb.setObjectName("resultThumb")
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setFixedSize(68, 68)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(3)

        self.title = QLabel("No match")
        self.title.setObjectName("resultTitle")
        self.subtitle = QLabel("-")
        self.subtitle.setObjectName("resultSubtitle")
        self.meta = QLabel("-")
        self.meta.setObjectName("resultMeta")
        self.meta.setWordWrap(True)
        self.pipeline = QLabel("-")
        self.pipeline.setObjectName("resultPipeline")

        text_col.addWidget(self.title)
        text_col.addWidget(self.subtitle)
        text_col.addWidget(self.meta)
        text_col.addWidget(self.pipeline)

        layout.addWidget(self.thumb)
        layout.addLayout(text_col, 1)

        right_col = QVBoxLayout()
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(6)
        right_col.setAlignment(Qt.AlignTop | Qt.AlignRight)

        self.score = QLabel("--")
        self.score.setObjectName("resultScore")
        self.distance = QLabel("--")
        self.distance.setObjectName("resultDistance")
        self.distance.setAlignment(Qt.AlignRight)

        right_col.addWidget(self.score)
        right_col.addWidget(self.distance)
        layout.addLayout(right_col)

    def set_result(self, result: dict, *, rank: int = 1) -> None:
        label = result.get("label") or "Unknown subject"
        score = result.get("score")
        distance = result.get("distance")
        pipeline = result.get("pipeline", "n/a")
        person_id = result.get("person_id", "")
        det_score = result.get("detection_score")
        face_index = int(result.get("face_index", 0)) + 1

        self.title.setText(label)
        self.subtitle.setText(f"Person ID: {person_id[:18]}")
        meta_parts = [f"Face {face_index}", f"Rank {rank}"]
        if det_score is not None:
            meta_parts.append(f"Det {det_score:.3f}")
        self.meta.setText("  |  ".join(meta_parts))
        self.pipeline.setText(f"Pipeline: {pipeline}")
        self.score.setText(f"{(score or 0.0) * 100:.1f}%")
        self.distance.setText(f"Distance {distance:.4f}" if distance is not None else "Distance --")


class LiveFaceLine(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("liveFaceLine")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)

        self.title = QLabel("Face 1")
        self.title.setObjectName("liveFaceTitle")
        self.meta = QLabel("-")
        self.meta.setObjectName("liveFaceMeta")
        self.meta.setWordWrap(True)
        text_col.addWidget(self.title)
        text_col.addWidget(self.meta)

        layout.addLayout(text_col, 1)

        self.status = QLabel("Unknown")
        self.status.setObjectName("liveFaceStatus")
        self.status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status)

    def set_face(self, face: dict) -> None:
        face_index = int(face.get("face_index", 0)) + 1
        label = face.get("label") or "Unknown"
        pipeline = str(face.get("pipeline", "-"))
        score = face.get("score")
        det_score = face.get("detection_score")
        quality = str(face.get("quality", "unknown")).lower()

        self.title.setText(f"Face {face_index} - {label}")
        parts = []
        if pipeline and pipeline != "-":
            parts.append(f"Pipeline: {pipeline}")
        if score is not None:
            parts.append(f"Score: {float(score):.4f}")
        if det_score is not None:
            parts.append(f"Det: {float(det_score):.3f}")
        self.meta.setText("  |  ".join(parts) if parts else "No technical data")

        status_map = {
            "match": ("Match", "ok"),
            "weak": ("Weak", "warn"),
            "unknown": ("Unknown", "error"),
        }
        text, state = status_map.get(quality, ("Unknown", "error"))
        self.status.setText(text)
        self.status.setProperty("state", state)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)


class PersonCard(QFrame):
    selected = Signal(str)
    deleteRequested = Signal(str)

    def __init__(self, person: dict, parent=None) -> None:
        super().__init__(parent)
        self.person = person
        self.person_id = person.get("id", "")
        self.setObjectName("personCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        heading = QLabel(person.get("label") or "Unnamed subject")
        heading.setObjectName("personCardTitle")
        meta = QLabel(f"ID: {self.person_id[:16]}")
        meta.setObjectName("personCardMeta")
        created = QLabel(f"Created: {str(person.get('created_at', ''))[:19].replace('T', ' ')}")
        created.setObjectName("personCardMeta")

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 8, 0, 0)
        button_row.setSpacing(8)
        profile_btn = ActionButton("Profile")
        delete_btn = ActionButton("Delete")
        profile_btn.clicked.connect(lambda: self.selected.emit(self.person_id))
        delete_btn.clicked.connect(lambda: self.deleteRequested.emit(self.person_id))
        button_row.addWidget(profile_btn)
        button_row.addWidget(delete_btn)

        layout.addWidget(heading)
        layout.addWidget(meta)
        layout.addWidget(created)
        layout.addStretch()
        layout.addLayout(button_row)


class ImagePreview(QLabel):
    def __init__(self, size: int = 160, parent=None) -> None:
        super().__init__(parent)
        self._size = size
        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignCenter)
        self.setObjectName("compactPreview")
        self.setText("No image")

    def load(self, path: str) -> None:
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.clear_preview()
            self.setText("Invalid")
            return
        scaled = pixmap.scaled(self._size, self._size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

    def clear_preview(self) -> None:
        self.clear()
        self.setText("No image")


def shorten_path(path: str, *, max_length: int = 60) -> str:
    if len(path) <= max_length:
        return path
    file_name = Path(path).name
    keep = max_length - len(file_name) - 4
    if keep <= 0:
        return file_name
    return f"{path[:keep]}...{file_name}"
