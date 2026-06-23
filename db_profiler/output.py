from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STABLE_TOP_LEVEL_KEYS = (
    "metadata",
    "tables",
    "relationships",
    "database_health",
    "query_generation_hints",
)


def empty_profile_document() -> dict[str, Any]:
    return {
        "metadata": {},
        "tables": {},
        "relationships": {
            "declared": [],
            "inferred_candidates": [],
            "normalized_join_candidates": [],
        },
        "database_health": {
            "warnings": [],
            "unavailable_checks": [],
        },
        "query_generation_hints": {
            "join_graph": [],
            "do_not_auto_join": [],
        },
    }


def write_profile_json(profile: dict[str, Any], output_path: Path) -> None:
    missing = [key for key in STABLE_TOP_LEVEL_KEYS if key not in profile]
    if missing:
        raise ValueError(f"Profile output is missing required top-level key(s): {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

