#!/usr/bin/env python3
"""Validate exported custom pipeline bundle before integration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


COMMON_ENV_KEYS = {
    "ENABLE_CUSTOM_PIPELINE",
    "CUSTOM_BACKEND",
    "CUSTOM_INDEX_PATH",
}

BACKEND_ENV_KEYS = {
    "onnx": {"ONNX_DETECTOR_PATH", "ONNX_EMBEDDER_PATH", "EMBEDDING_DIM"},
    "torch": {"TORCH_MODEL_PATH", "TORCH_MODEL_ARCH", "TORCH_INPUT_SIZE", "TORCH_DEVICE"},
    "insightface": {"MODEL_NAME", "EMBEDDING_DIM"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate custom pipeline bundle artifacts and config.")
    parser.add_argument(
        "--bundle_dir",
        type=Path,
        required=True,
        help="Path to exported bundle directory (contains .env.custom).",
    )
    parser.add_argument(
        "--require_manifest",
        action="store_true",
        help="Fail if manifest.json is missing.",
    )
    parser.add_argument(
        "--check_onnx_output",
        action="store_true",
        help="Try to open ONNX embedder and verify output dimension.",
    )
    return parser.parse_args()


def parse_env_file(path: Path) -> dict[str, str]:
    env_map: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            env_map[key] = value.strip()
    return env_map


def rel_from_app_path(value: str) -> Path | None:
    prefix = "/app/model_bundle/"
    if not value.startswith(prefix):
        return None
    return Path(value[len(prefix) :])


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest.json must contain a JSON object")
    return payload


def build_requirements_filename(backend: str) -> str:
    return f"requirements-ml-{backend}.txt"


def maybe_check_onnx_dim(embedder_path: Path, expected_dim: int) -> tuple[bool, str]:
    try:
        import onnxruntime as ort
    except ImportError:
        return True, "onnxruntime not installed, skipped ONNX output-dim check"

    try:
        session = ort.InferenceSession(str(embedder_path), providers=["CPUExecutionProvider"])
    except Exception as exc:  # pragma: no cover
        return False, f"cannot open ONNX embedder: {exc}"

    outputs = session.get_outputs()
    if not outputs:
        return False, "embedder has no outputs"

    shape = outputs[0].shape
    if not shape:
        return False, "embedder output shape is empty"

    last_dim = shape[-1]
    if isinstance(last_dim, str) or last_dim is None:
        return True, f"embedder output shape is dynamic ({shape}), skipped strict dim compare"

    try:
        last_dim_int = int(last_dim)
    except Exception:  # pragma: no cover
        return True, f"embedder output shape is non-integer ({shape}), skipped strict dim compare"

    if last_dim_int != expected_dim:
        return False, f"embedder output last dim {last_dim_int} != EMBEDDING_DIM {expected_dim}"
    return True, f"embedder output dimension check passed ({last_dim_int})"


def main() -> None:
    args = parse_args()
    bundle_dir = args.bundle_dir.expanduser().resolve()

    errors: list[str] = []
    notes: list[str] = []

    if not bundle_dir.exists() or not bundle_dir.is_dir():
        raise SystemExit(f"bundle_dir does not exist: {bundle_dir}")

    env_path = bundle_dir / ".env.custom"
    if not env_path.exists():
        raise SystemExit(f"missing .env.custom: {env_path}")

    env_map = parse_env_file(env_path)
    missing_common = sorted(COMMON_ENV_KEYS - env_map.keys())
    if missing_common:
        errors.append(f"missing common env keys: {', '.join(missing_common)}")

    backend = env_map.get("CUSTOM_BACKEND")
    if backend not in BACKEND_ENV_KEYS:
        errors.append(f"unsupported CUSTOM_BACKEND: {backend}")
    else:
        missing_backend = sorted(BACKEND_ENV_KEYS[backend] - env_map.keys())
        if missing_backend:
            errors.append(f"missing backend env keys for {backend}: {', '.join(missing_backend)}")

    if env_map.get("ENABLE_CUSTOM_PIPELINE", "").lower() != "true":
        errors.append("ENABLE_CUSTOM_PIPELINE must be true")

    index_path = env_map.get("CUSTOM_INDEX_PATH")
    if index_path:
        rel = rel_from_app_path(index_path)
        if rel is None:
            errors.append("CUSTOM_INDEX_PATH must start with /app/model_bundle/")
        else:
            local_index_dir = bundle_dir / rel
            if not local_index_dir.exists():
                errors.append(f"CUSTOM_INDEX_PATH points to missing dir: {local_index_dir}")
            else:
                for required_index_file in ("faiss.index", "meta.json"):
                    if not (local_index_dir / required_index_file).exists():
                        errors.append(f"missing index file: {local_index_dir / required_index_file}")

    onnx_embedder_local: Path | None = None
    if backend == "onnx":
        for key in ("ONNX_DETECTOR_PATH", "ONNX_EMBEDDER_PATH"):
            value = env_map.get(key, "")
            rel = rel_from_app_path(value)
            if rel is None:
                errors.append(f"{key} must start with /app/model_bundle/")
                continue
            local_path = bundle_dir / rel
            if not local_path.exists():
                errors.append(f"{key} points to missing file: {local_path}")
            elif key == "ONNX_EMBEDDER_PATH":
                onnx_embedder_local = local_path

        try:
            emb_dim = int(env_map.get("EMBEDDING_DIM", ""))
            if emb_dim <= 0:
                raise ValueError("non-positive")
        except Exception:
            errors.append("EMBEDDING_DIM must be a positive integer")
            emb_dim = 0

        if args.check_onnx_output and onnx_embedder_local and emb_dim > 0:
            ok, msg = maybe_check_onnx_dim(onnx_embedder_local, emb_dim)
            if ok:
                notes.append(msg)
            else:
                errors.append(msg)

    if backend == "torch":
        arch = env_map.get("TORCH_MODEL_ARCH")
        if arch not in {"ir18", "ir34", "ir50", "ir100"}:
            errors.append("TORCH_MODEL_ARCH must be one of: ir18, ir34, ir50, ir100")

        model_path = env_map.get("TORCH_MODEL_PATH", "")
        rel = rel_from_app_path(model_path)
        if rel is None:
            errors.append("TORCH_MODEL_PATH must start with /app/model_bundle/")
        else:
            local_path = bundle_dir / rel
            if not local_path.exists():
                errors.append(f"TORCH_MODEL_PATH points to missing file: {local_path}")

    if backend == "insightface":
        try:
            emb_dim = int(env_map.get("EMBEDDING_DIM", ""))
            if emb_dim <= 0:
                raise ValueError("non-positive")
        except Exception:
            errors.append("EMBEDDING_DIM must be a positive integer for insightface")

    if backend in BACKEND_ENV_KEYS:
        requirements_path = bundle_dir / build_requirements_filename(backend)
        if not requirements_path.exists():
            errors.append(f"missing backend runtime requirements file: {requirements_path}")

    smoke_script_path = bundle_dir / "smoke_runtime.py"
    if not smoke_script_path.exists():
        errors.append(f"missing runtime smoke script: {smoke_script_path}")

    smoke_inputs_dir = bundle_dir / "smoke_inputs"
    if smoke_inputs_dir.exists():
        expected_smoke_files = {"single_face.jpg", "no_face.png", "multiple_faces.jpg"}
        existing = {p.name for p in smoke_inputs_dir.iterdir() if p.is_file()}
        missing_smoke = sorted(expected_smoke_files - existing)
        if missing_smoke:
            notes.append(
                "smoke_inputs is incomplete, provide manually: " + ", ".join(missing_smoke)
            )
    else:
        notes.append("smoke_inputs directory is missing, runtime smoke tests need manual images")

    manifest_path = bundle_dir / "manifest.json"
    if args.require_manifest and not manifest_path.exists():
        errors.append(f"missing manifest.json: {manifest_path}")

    if manifest_path.exists():
        try:
            manifest = load_manifest(manifest_path)
        except Exception as exc:
            errors.append(f"invalid manifest.json: {exc}")
        else:
            manifest_backend = manifest.get("backend")
            if backend and manifest_backend != backend:
                errors.append(
                    f"backend mismatch: .env has {backend}, manifest has {manifest_backend}"
                )

            manifest_env = manifest.get("env")
            if isinstance(manifest_env, dict):
                for key, value in manifest_env.items():
                    if key in env_map and str(env_map[key]) != str(value):
                        errors.append(f"env mismatch for {key}: manifest={value} env={env_map[key]}")

            manifest_files = manifest.get("files")
            if isinstance(manifest_files, list):
                for idx, item in enumerate(manifest_files):
                    if not isinstance(item, dict):
                        errors.append(f"manifest files[{idx}] is not an object")
                        continue
                    rel_path = item.get("path")
                    sha_expected = item.get("sha256")
                    if not isinstance(rel_path, str) or not isinstance(sha_expected, str):
                        errors.append(f"manifest files[{idx}] must contain string path + sha256")
                        continue

                    local_path = bundle_dir / Path(rel_path)
                    if not local_path.exists():
                        errors.append(f"manifest file missing: {local_path}")
                        continue

                    actual_sha = sha256_file(local_path)
                    if actual_sha != sha_expected:
                        errors.append(
                            f"checksum mismatch for {rel_path}: expected {sha_expected}, got {actual_sha}"
                        )
            else:
                notes.append("manifest has no files[] section, skipped checksum checks from manifest")

    checksums_path = bundle_dir / "checksums.sha256"
    if checksums_path.exists():
        for line_num, raw in enumerate(checksums_path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.strip()
            if not line:
                continue
            if " *" not in line:
                errors.append(f"invalid checksums.sha256 line {line_num}: {line}")
                continue
            expected, rel_path = line.split(" *", 1)
            target = bundle_dir / Path(rel_path)
            if not target.exists():
                errors.append(f"checksums target missing: {target}")
                continue
            actual = sha256_file(target)
            if actual != expected:
                errors.append(
                    f"checksums mismatch for {rel_path}: expected {expected}, got {actual}"
                )
    else:
        notes.append("checksums.sha256 missing, skipped checksum file verification")

    if errors:
        print("VALIDATION FAILED")
        for item in errors:
            print(f"- {item}")
        if notes:
            print("\nNotes:")
            for note in notes:
                print(f"- {note}")
        raise SystemExit(1)

    print("VALIDATION PASSED")
    print(f"- bundle_dir: {bundle_dir}")
    if backend:
        print(f"- backend: {backend}")
    if notes:
        print("Notes:")
        for note in notes:
            print(f"- {note}")


if __name__ == "__main__":
    main()
