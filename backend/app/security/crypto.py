from __future__ import annotations

import base64
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.core.config import settings

_PAYLOAD_PREFIX = b"ENC1"
_NONCE_SIZE = 12
_TEST_KEY = base64.urlsafe_b64encode(b"\x11" * 32).decode("ascii")


def _decode_key(raw_key: str, *, field_name: str) -> bytes:
    source = raw_key or (_TEST_KEY if settings.testing else "")
    if not source:
        raise RuntimeError(f"{field_name} must be set")

    padded = source + ("=" * (-len(source) % 4))
    try:
        decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"{field_name} must be a urlsafe base64-encoded 32-byte key") from exc

    if len(decoded) != 32:
        raise RuntimeError(f"{field_name} must decode to exactly 32 bytes")
    return decoded


def _aesgcm(raw_key: str, *, field_name: str) -> AESGCM:
    return AESGCM(_decode_key(raw_key, field_name=field_name))


def _encrypt_bytes(raw: bytes, *, raw_key: str, field_name: str, aad: bytes) -> bytes:
    nonce = os.urandom(_NONCE_SIZE)
    cipher = _aesgcm(raw_key, field_name=field_name)
    ciphertext = cipher.encrypt(nonce, raw, aad)
    return _PAYLOAD_PREFIX + nonce + ciphertext


def _decrypt_bytes(blob: bytes, *, raw_key: str, field_name: str, aad: bytes) -> bytes:
    if not blob.startswith(_PAYLOAD_PREFIX):
        # Backward compatibility for legacy unencrypted rows / snapshots.
        return blob
    nonce = blob[len(_PAYLOAD_PREFIX):len(_PAYLOAD_PREFIX) + _NONCE_SIZE]
    ciphertext = blob[len(_PAYLOAD_PREFIX) + _NONCE_SIZE:]
    cipher = _aesgcm(raw_key, field_name=field_name)
    return cipher.decrypt(nonce, ciphertext, aad)


def encrypt_embedding_payload(raw: bytes) -> bytes:
    return _encrypt_bytes(
        raw,
        raw_key=settings.data_encryption_key,
        field_name="DATA_ENCRYPTION_KEY",
        aad=b"embedding-vector",
    )


def decrypt_embedding_payload(blob: bytes) -> bytes:
    return _decrypt_bytes(
        blob,
        raw_key=settings.data_encryption_key,
        field_name="DATA_ENCRYPTION_KEY",
        aad=b"embedding-vector",
    )


def encrypt_snapshot_payload(raw: bytes) -> bytes:
    return _encrypt_bytes(
        raw,
        raw_key=settings.snapshot_encryption_key,
        field_name="SNAPSHOT_ENCRYPTION_KEY",
        aad=b"faiss-snapshot",
    )


def decrypt_snapshot_payload(blob: bytes) -> bytes:
    return _decrypt_bytes(
        blob,
        raw_key=settings.snapshot_encryption_key,
        field_name="SNAPSHOT_ENCRYPTION_KEY",
        aad=b"faiss-snapshot",
    )
