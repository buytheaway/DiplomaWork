from __future__ import annotations

from types import SimpleNamespace

from scripts.import_bundle import (
    clean_person_label,
    default_model_name_for_pipeline,
    resolve_target_index_path,
)


def test_default_model_name_for_pipeline() -> None:
    assert default_model_name_for_pipeline("pretrained") == "onnx_w600k_r50"
    assert default_model_name_for_pipeline("custom") == "torch_ir50"


def test_clean_person_label() -> None:
    assert clean_person_label("AC_Test_Person") == "Test Person"
    assert clean_person_label("Plain_Name") == "Plain Name"


def test_custom_pipeline_uses_custom_index_path() -> None:
    settings = SimpleNamespace(
        pretrained_index_path="backend/data/index/pretrained.faiss",
        custom_index_path="backend/data/index/custom.faiss",
    )

    resolved = resolve_target_index_path(settings, "custom", None)

    assert resolved.as_posix().endswith("backend/data/index/custom.faiss")


def test_index_path_override_wins() -> None:
    settings = SimpleNamespace(
        pretrained_index_path="backend/data/index/pretrained.faiss",
        custom_index_path="backend/data/index/custom.faiss",
    )

    resolved = resolve_target_index_path(settings, "custom", "tmp/override.faiss")

    assert resolved.as_posix().endswith("tmp/override.faiss")
