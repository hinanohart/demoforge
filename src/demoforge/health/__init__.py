"""Demo-health sidecar: per-episode triage telemetry as JSON Lines.

The sidecar reports the *transform* the re-timer applied (before/after limit violations, jerk,
path deviation, locked contact segments) so a human can triage which demos to keep. It is
deliberately NOT an aggregate quality score and NOT a predictor of downstream training outcomes — see
:class:`demoforge.ir.DemoHealth`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from ..ir import DemoHealth

__all__ = ["SIDECAR_SCHEMA", "write_sidecar", "validate_record"]

# JSON Schema for one sidecar record (used by tests and `validate_record`).
SIDECAR_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": [
        "episode_index",
        "source",
        "backend",
        "frames_in",
        "frames_out",
        "limit_violations_before",
        "limit_violations_after",
        "max_jerk_before",
        "max_jerk_after",
        "path_deviation_max",
        "retime_outcome",
        "triage",
    ],
    "properties": {
        "episode_index": {"type": "integer"},
        "source": {"type": "string"},
        "backend": {"type": "string"},
        "frames_in": {"type": "integer", "minimum": 0},
        "frames_out": {"type": "integer", "minimum": 0},
        "limit_violations_before": {"type": "integer", "minimum": 0},
        "limit_violations_after": {"type": "integer", "minimum": 0},
        "max_jerk_before": {"type": "number"},
        "max_jerk_after": {"type": "number"},
        "path_deviation_max": {"type": "number", "minimum": 0},
        "contact_segments_locked": {"type": "integer", "minimum": 0},
        "retime_outcome": {
            "type": "string",
            "enum": ["retimed", "infeasible_passthrough"],
        },
        "triage": {
            "type": "object",
            "required": ["flags", "suggestion"],
            "properties": {
                "flags": {"type": "array", "items": {"type": "string"}},
                "suggestion": {"type": "string"},
            },
        },
    },
    # By construction the sidecar carries no single aggregate quality score.
    "not": {"required": ["quality_score"]},
}


def validate_record(record: dict) -> None:
    """Raise ``jsonschema.ValidationError`` if ``record`` does not match :data:`SIDECAR_SCHEMA`."""
    import jsonschema

    jsonschema.validate(record, SIDECAR_SCHEMA)


def write_sidecar(path: str | Path, records: Iterable[DemoHealth]) -> Path:
    """Write health records as JSON Lines (one validated JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            d = rec.to_dict()
            validate_record(d)
            f.write(json.dumps(d, sort_keys=True))
            f.write("\n")
    return path
