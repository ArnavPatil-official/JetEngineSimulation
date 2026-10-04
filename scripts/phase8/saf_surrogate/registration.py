"""Standard-library registration, integrity and write-once artifact helpers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

REGISTRATION = "docs/phase8_saf_surrogate_registration.json"
REGISTRATION_ID = "P8-S-20261004"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else json_bytes(value)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def append_progress(path, event):
    """Append only inside the caller's verified exclusive run lease."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write((json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def contained(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Artifact paths must stay inside the registered attempt")
    result = (root / relative).resolve()
    if result != root and root not in result.parents:
        raise ValueError("Artifact path escapes the registered attempt")
    return result


def load_registration(root, path=REGISTRATION):
    path = Path(root) / path
    doc = read_json(path)
    if doc["id"] != REGISTRATION_ID or doc["state"] != "PROSPECTIVE_REGISTRATION_NOT_ARMED":
        raise ValueError("Foreign or draft SAF registration")
    if doc["sampling"]["train_sizes"] != [64, 128, 256, 512, 1024, 2048, 4096]:
        raise ValueError("Unregistered learning budget")
    if doc["model"]["seeds"] != [42, 43, 44] or doc["model"]["output_dimension"] != 494:
        raise ValueError("Unregistered model/seed contract")
    if not doc["provenance"]["gate_API"]["require_g0"]:
        raise ValueError("G0 evidence cannot be bypassed")
    return doc, sha256_file(path)


def artifact_hashes(output, paths):
    return {name: sha256_file(contained(output, name)) for name in paths}


def verify_artifacts(output, hashes):
    for name, expected in hashes.items():
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"Missing artifact identity: {name}")
        if sha256_file(contained(output, name)) != expected:
            raise ValueError(f"Artifact drift: {name}")


def verify_named_prerequisite(root, reg, binary_sha256):
    """Read provenance only; never open a named numeric target here."""
    from scripts.phase8 import scientific_workflow_gate as gate

    path = reg["sole_score"]["named_registration"]
    prerequisite = read_json(Path(root) / path)
    output = Path(root) / "outputs/phase7/p73_a1_cpp_20261004"
    expected_registration = sha256_file(Path(root) / path)
    # The shared validator must verify raw record semantics, source identity and
    # output hashes without granting this function numeric-target access.
    validator = getattr(gate, "validate_consumer_terminal", None)
    if validator is None:
        raise RuntimeError("Shared consumer-terminal validator is unavailable")
    metadata = validator(root, path, output, expected_binary_sha256=binary_sha256)
    if metadata["core_sha256"] != binary_sha256:
        raise ValueError("Named prerequisite used a different core")
    return {"registration": path, "registration_sha256": expected_registration,
            "terminal_sha256": sha256_file(output / "terminal.json"),
            "environment_sha256": sha256_file(output / "environment.json"),
            "output": str(output), "id": prerequisite.get("id", prerequisite.get("registration_id"))}
