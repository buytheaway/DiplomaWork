from __future__ import annotations

from typing import Any

from PySide6.QtGui import QImage, QPixmap

from app.ui.live_geometry import encoded_frame_geometry

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency check
    cv2 = None


def encode_frame_for_upload(
    frame: Any,
    *,
    max_width: int,
    jpeg_quality: int,
) -> tuple[bytes, tuple[float, float]] | None:
    if cv2 is None:
        return None
    height, width = frame.shape[:2]
    encoded_w, encoded_h, bbox_scale = encoded_frame_geometry(
        width=width,
        height=height,
        max_width=max_width,
    )
    if encoded_w != width or encoded_h != height:
        frame = cv2.resize(frame, (encoded_w, encoded_h), interpolation=cv2.INTER_AREA)
    quality = min(max(jpeg_quality, 40), 95)
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return buffer.tobytes(), bbox_scale


def encode_image_file_for_upload(
    path: str,
    *,
    max_width: int,
    jpeg_quality: int,
) -> tuple[bytes, tuple[float, float]] | None:
    if cv2 is None:
        return None
    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    if frame is None:
        return None
    return encode_frame_for_upload(
        frame,
        max_width=max_width,
        jpeg_quality=jpeg_quality,
    )


def frame_to_pixmap(frame: Any, *, max_width: int = 640) -> QPixmap | None:
    if cv2 is None:
        return None
    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        frame = cv2.resize(
            frame,
            (max_width, int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
        height, width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _height, _width, channels = rgb.shape
    bytes_per_line = channels * width
    image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(image)


def draw_live_annotations(frame: Any, annotations: list[dict[str, Any]]) -> Any:
    if cv2 is None or not annotations:
        return frame
    for annotation in annotations:
        bbox = annotation.get("face_bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(max(0, value)) for value in bbox]
        quality = str(annotation.get("quality", "unknown"))
        color = overlay_color(quality)
        label = annotation.get("label") or "Unknown"
        score = annotation.get("score")
        pipeline = annotation.get("pipeline")
        parts = [label]
        if pipeline:
            parts.append(str(pipeline))
        if score is not None:
            parts.append(f"{float(score):.3f}")
        text = " | ".join(parts)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(
            frame,
            (x1, max(0, y1 - 26)),
            (min(frame.shape[1] - 1, x1 + max(120, len(text) * 7)), y1),
            color,
            -1,
        )
        cv2.putText(
            frame,
            text,
            (x1 + 4, max(14, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (4, 16, 23),
            1,
            cv2.LINE_AA,
        )
    return frame


def overlay_color(quality: str) -> tuple[int, int, int]:
    if quality == "match":
        return (55, 243, 187)
    if quality == "weak":
        return (115, 204, 255)
    return (136, 107, 255)
