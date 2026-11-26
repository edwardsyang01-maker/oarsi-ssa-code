import hashlib
import json
import os
import uuid
from pathlib import Path


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def project_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("preprocessing outputs/lists must stay inside the drai data directory project")
    return path


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def sealed_record(payload):
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode()
    return {"payload": payload, "payload_sha256": hashlib.sha256(encoded).hexdigest()}


def read_sealed_record(path):
    record = json.loads(Path(path).read_text())
    if sealed_record(record["payload"])["payload_sha256"] != record["payload_sha256"]:
        raise ValueError("preprocessing result fingerprint mismatch")
    return record["payload"]
