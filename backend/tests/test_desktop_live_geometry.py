from __future__ import annotations

from desktop.app.ui.live_geometry import encoded_frame_geometry, scale_bbox


def test_encoded_frame_geometry_tracks_api_to_source_scale():
    width, height, scale = encoded_frame_geometry(width=1280, height=720, max_width=640)

    assert (width, height) == (640, 360)
    assert scale == (2.0, 2.0)


def test_scale_bbox_maps_api_bbox_back_to_source_frame():
    bbox = scale_bbox([10.0, 20.0, 100.0, 200.0], (2.0, 2.0))

    assert bbox == [20.0, 40.0, 200.0, 400.0]


def test_scale_bbox_rejects_missing_or_malformed_bbox():
    assert scale_bbox(None, (2.0, 2.0)) is None
    assert scale_bbox([1.0, 2.0], (2.0, 2.0)) is None
