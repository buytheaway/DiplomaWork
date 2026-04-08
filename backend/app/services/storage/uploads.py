from __future__ import annotations

from fastapi import UploadFile


class UploadValidationError(ValueError):
    pass


def _format_byte_limit(max_bytes: int) -> str:
    if max_bytes >= 1024 * 1024:
        return f"{max_bytes // (1024 * 1024)} MB"
    if max_bytes >= 1024:
        return f"{max_bytes // 1024} KB"
    return f"{max_bytes} bytes"


def allowed_content_types(raw_value: str) -> set[str]:
    return {
        item.strip().lower()
        for item in raw_value.split(",")
        if item.strip()
    }


async def read_image_upload(
    file: UploadFile,
    *,
    max_bytes: int,
    allowed_types: set[str],
) -> bytes:
    content_type = (file.content_type or "").strip().lower()
    if content_type and content_type not in allowed_types:
        raise UploadValidationError(
            "Unsupported file type. Use JPG, PNG, BMP or WEBP."
        )

    chunks: list[bytes] = []
    total = 0
    chunk_size = 1024 * 1024

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise UploadValidationError(
                f"Uploaded file is too large. Limit is {_format_byte_limit(max_bytes)}."
            )
        chunks.append(chunk)

    if total == 0:
        raise UploadValidationError("Uploaded file is empty.")

    return b"".join(chunks)
