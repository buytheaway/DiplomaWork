from __future__ import annotations

from collections.abc import Sequence


def encoded_frame_geometry(
    width: int,
    height: int,
    max_width: int,
) -> tuple[int, int, tuple[float, float]]:
    """Return encoded size and scale from encoded API image back to source frame."""
    safe_max_width = max(160, max_width)
    if width <= safe_max_width:
        return width, height, (1.0, 1.0)

    encode_scale = safe_max_width / width
    encoded_height = max(1, int(height * encode_scale))
    return safe_max_width, encoded_height, (
        width / safe_max_width,
        height / encoded_height,
    )


def scale_bbox(
    bbox: Sequence[float] | None,
    scale: tuple[float, float],
) -> list[float] | None:
    if bbox is None or len(bbox) != 4:
        return None

    sx, sy = scale
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
