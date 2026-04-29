#!/usr/bin/env python3
"""Runtime smoke tests for custom face pipeline in a target backend."""
from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import uuid
from pathlib import Path
from typing import Any
from urllib import error, request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run health + runtime smoke checks for custom pipeline integration."
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Target backend base URL.",
    )
    parser.add_argument(
        "--health_path",
        type=str,
        default="/v1/health",
        help="Health endpoint path or full URL.",
    )
    parser.add_argument(
        "--embed_path",
        type=str,
        default="",
        help="Embedding endpoint path or full URL. Required unless --skip_runtime_checks.",
    )
    parser.add_argument(
        "--skip_runtime_checks",
        action="store_true",
        help="Run only /v1/health checks.",
    )
    parser.add_argument(
        "--request_mode",
        type=str,
        choices=["multipart", "json_base64"],
        default="multipart",
        help="How image is sent to embed endpoint.",
    )
    parser.add_argument(
        "--input_key",
        type=str,
        default="image",
        help="Multipart field name or JSON key (for json_base64 mode).",
    )
    parser.add_argument(
        "--single_face_image",
        type=Path,
        default=None,
        help="Path to image containing exactly one face.",
    )
    parser.add_argument(
        "--no_face_image",
        type=Path,
        default=None,
        help="Path to image containing no face.",
    )
    parser.add_argument(
        "--multiple_faces_image",
        type=Path,
        default=None,
        help="Path to image containing multiple faces.",
    )
    parser.add_argument(
        "--expected_embedding_dim",
        type=int,
        default=512,
        help="Expected embedding vector size.",
    )
    parser.add_argument(
        "--stability_l2_tolerance",
        type=float,
        default=1e-5,
        help="Max L2 distance between two embeddings from the same image.",
    )
    parser.add_argument(
        "--l2_norm_tolerance",
        type=float,
        default=1e-3,
        help="Allowed absolute error for expected L2 norm ~= 1.",
    )
    parser.add_argument(
        "--skip_l2_norm_check",
        action="store_true",
        help="Skip L2-normalization check for embeddings.",
    )
    parser.add_argument(
        "--no_face_code",
        type=str,
        default="no_face",
        help="Expected error/status code for no-face input.",
    )
    parser.add_argument(
        "--multiple_faces_code",
        type=str,
        default="multiple_faces",
        help="Expected error/status code for multiple-faces input.",
    )
    parser.add_argument(
        "--timeout_sec",
        type=float,
        default=30.0,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="Extra HTTP header in KEY:VALUE format. Can be passed multiple times.",
    )
    return parser.parse_args()


def build_url(base_url: str, path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return path_or_url
    if not path_or_url:
        return base_url.rstrip("/")
    if not path_or_url.startswith("/"):
        path_or_url = "/" + path_or_url
    return base_url.rstrip("/") + path_or_url


def parse_headers(raw_headers: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for item in raw_headers:
        if ":" not in item:
            raise ValueError(f"Header must be KEY:VALUE, got: {item}")
        key, value = item.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Header key is empty: {item}")
        headers[key] = value.strip()
    return headers


def request_json(
    method: str,
    url: str,
    body: bytes | None,
    headers: dict[str, str],
    timeout_sec: float,
) -> tuple[int, Any]:
    req = request.Request(url=url, data=body, method=method.upper())
    for key, value in headers.items():
        req.add_header(key, value)

    status_code: int
    raw: bytes
    try:
        with request.urlopen(req, timeout=timeout_sec) as response:
            status_code = int(response.status)
            raw = response.read()
    except error.HTTPError as exc:
        status_code = int(exc.code)
        raw = exc.read()
    except error.URLError as exc:
        return 0, {"error": f"connection_error: {exc}"}

    text = raw.decode("utf-8", errors="replace")
    if not text.strip():
        return status_code, {}
    try:
        return status_code, json.loads(text)
    except json.JSONDecodeError:
        return status_code, {"raw_text": text}


def build_multipart_body(input_key: str, image_path: Path) -> tuple[bytes, str]:
    boundary = f"----CustomPipelineBoundary{uuid.uuid4().hex}"
    mime_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
    file_name = image_path.name
    image_bytes = image_path.read_bytes()

    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{input_key}"; filename="{file_name}"\r\n'
        f"Content-Type: {mime_type}\r\n\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = head + image_bytes + tail
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def post_image(
    url: str,
    input_key: str,
    image_path: Path,
    request_mode: str,
    headers: dict[str, str],
    timeout_sec: float,
) -> tuple[int, Any]:
    req_headers = dict(headers)
    if request_mode == "multipart":
        body, content_type = build_multipart_body(input_key=input_key, image_path=image_path)
        req_headers["Content-Type"] = content_type
        return request_json("POST", url, body, req_headers, timeout_sec)

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    body_json = json.dumps({input_key: encoded}, ensure_ascii=False).encode("utf-8")
    req_headers["Content-Type"] = "application/json"
    return request_json("POST", url, body_json, req_headers, timeout_sec)


def get_by_path(payload: Any, path: list[str]) -> Any:
    current = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def is_numeric_list(value: Any) -> bool:
    return isinstance(value, list) and value and all(isinstance(x, (int, float)) for x in value)


def extract_embedding(payload: Any) -> list[float] | None:
    if is_numeric_list(payload):
        return [float(x) for x in payload]

    paths = [
        ["embedding"],
        ["vector"],
        ["data", "embedding"],
        ["data", "vector"],
        ["result", "embedding"],
        ["result", "vector"],
    ]

    if isinstance(payload, dict):
        for path in paths:
            value = get_by_path(payload, path)
            if is_numeric_list(value):
                return [float(x) for x in value]
            if isinstance(value, list) and value and is_numeric_list(value[0]):
                return [float(x) for x in value[0]]

        embeddings = payload.get("embeddings")
        if is_numeric_list(embeddings):
            return [float(x) for x in embeddings]
        if isinstance(embeddings, list) and embeddings and is_numeric_list(embeddings[0]):
            return [float(x) for x in embeddings[0]]

    return None


def extract_code(payload: Any) -> str | None:
    keys = {"code", "error_code", "status", "error", "detail"}
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in payload.values():
            found = extract_code(value)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = extract_code(item)
            if found:
                return found
    return None


def l2_norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def l2_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


def ensure_file(path: Path | None, label: str) -> Path:
    if path is None:
        raise FileNotFoundError(f"{label} is not set")
    resolved = path.expanduser().resolve()
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def print_result(ok: bool, title: str, details: str) -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {title}: {details}")
    return ok


def check_health(health_payload: Any) -> tuple[bool, str]:
    if not isinstance(health_payload, dict):
        return False, f"health payload is not object: {health_payload}"

    available = health_payload.get("available_pipelines")
    if not isinstance(available, list):
        return False, f"available_pipelines is not list: {available}"
    if "custom" not in available:
        return False, f"custom not in available_pipelines: {available}"

    unavailable = health_payload.get("unavailable_pipelines")
    if unavailable != {}:
        return False, f"unavailable_pipelines expected {{}}, got: {unavailable}"

    return True, f"available_pipelines={available}, unavailable_pipelines={unavailable}"


def main() -> None:
    args = parse_args()

    headers = parse_headers(args.header)
    health_url = build_url(args.base_url, args.health_path)
    embed_url = build_url(args.base_url, args.embed_path) if args.embed_path else ""

    overall_ok = True

    status_code, health_payload = request_json(
        method="GET",
        url=health_url,
        body=None,
        headers=headers,
        timeout_sec=args.timeout_sec,
    )
    health_ok, health_details = check_health(health_payload)
    overall_ok &= print_result(
        health_ok and 200 <= status_code < 300,
        "Health",
        f"http_status={status_code}, {health_details}",
    )

    if args.skip_runtime_checks:
        if overall_ok:
            print("SMOKE TEST PASSED (health-only mode)")
            raise SystemExit(0)
        print("SMOKE TEST FAILED")
        raise SystemExit(1)

    if not embed_url:
        print("[FAIL] Runtime: --embed_path is required unless --skip_runtime_checks is set")
        raise SystemExit(1)

    try:
        single_face_image = ensure_file(args.single_face_image, "single_face_image")
        no_face_image = ensure_file(args.no_face_image, "no_face_image")
        multiple_faces_image = ensure_file(args.multiple_faces_image, "multiple_faces_image")
    except FileNotFoundError as exc:
        print(f"[FAIL] Runtime inputs: {exc}")
        raise SystemExit(1)

    status1, payload1 = post_image(
        url=embed_url,
        input_key=args.input_key,
        image_path=single_face_image,
        request_mode=args.request_mode,
        headers=headers,
        timeout_sec=args.timeout_sec,
    )
    emb1 = extract_embedding(payload1)
    ok_single = status1 < 400 and emb1 is not None
    details_single = f"http_status={status1}, embedding_found={emb1 is not None}"
    if emb1 is not None:
        details_single += f", dim={len(emb1)}"
    overall_ok &= print_result(ok_single, "Single Face -> Embedding", details_single)

    if emb1 is not None:
        dim_ok = len(emb1) == args.expected_embedding_dim
        overall_ok &= print_result(
            dim_ok,
            "Embedding Dimension",
            f"expected={args.expected_embedding_dim}, got={len(emb1)}",
        )

        if not args.skip_l2_norm_check:
            norm_value = l2_norm(emb1)
            norm_ok = abs(norm_value - 1.0) <= args.l2_norm_tolerance
            overall_ok &= print_result(
                norm_ok,
                "Embedding L2 Norm",
                f"norm={norm_value:.8f}, tolerance={args.l2_norm_tolerance}",
            )

    status2, payload2 = post_image(
        url=embed_url,
        input_key=args.input_key,
        image_path=single_face_image,
        request_mode=args.request_mode,
        headers=headers,
        timeout_sec=args.timeout_sec,
    )
    emb2 = extract_embedding(payload2)
    stab_ok = emb1 is not None and emb2 is not None and len(emb1) == len(emb2)
    stab_details = f"http_status_1={status1}, http_status_2={status2}"
    if stab_ok:
        dist = l2_distance(emb1, emb2)
        stab_ok = dist <= args.stability_l2_tolerance
        stab_details += (
            f", l2_distance={dist:.10f}, tolerance={args.stability_l2_tolerance}"
        )
    else:
        stab_details += ", embedding extraction failed for stability check"
    overall_ok &= print_result(stab_ok, "Embedding Stability", stab_details)

    status_no_face, payload_no_face = post_image(
        url=embed_url,
        input_key=args.input_key,
        image_path=no_face_image,
        request_mode=args.request_mode,
        headers=headers,
        timeout_sec=args.timeout_sec,
    )
    no_face_code = extract_code(payload_no_face)
    no_face_ok = (
        no_face_code is not None and no_face_code.lower() == args.no_face_code.lower()
    )
    overall_ok &= print_result(
        no_face_ok,
        "No Face Handling",
        f"http_status={status_no_face}, expected_code={args.no_face_code}, got_code={no_face_code}",
    )

    status_multi, payload_multi = post_image(
        url=embed_url,
        input_key=args.input_key,
        image_path=multiple_faces_image,
        request_mode=args.request_mode,
        headers=headers,
        timeout_sec=args.timeout_sec,
    )
    multiple_code = extract_code(payload_multi)
    multiple_ok = (
        multiple_code is not None
        and multiple_code.lower() == args.multiple_faces_code.lower()
    )
    overall_ok &= print_result(
        multiple_ok,
        "Multiple Faces Handling",
        f"http_status={status_multi}, expected_code={args.multiple_faces_code}, got_code={multiple_code}",
    )

    if overall_ok:
        print("SMOKE TEST PASSED")
        raise SystemExit(0)

    print("SMOKE TEST FAILED")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
