import json
from pathlib import Path

from db_profiler.output import STABLE_TOP_LEVEL_KEYS, empty_profile_document, write_profile_json


def test_empty_profile_document_has_stable_top_level_keys() -> None:
    document = empty_profile_document()

    assert tuple(document.keys()) == STABLE_TOP_LEVEL_KEYS
    assert document["relationships"]["declared"] == []
    assert document["database_health"]["warnings"] == []
    assert document["query_generation_hints"]["join_graph"] == []


def test_write_profile_json_creates_parent_directories(tmp_path: Path) -> None:
    document = empty_profile_document()
    document["metadata"] = {"run_id": "test"}
    output_path = tmp_path / "nested" / "profile.json"

    write_profile_json(document, output_path)

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["metadata"]["run_id"] == "test"
    assert tuple(loaded.keys()) == tuple(sorted(STABLE_TOP_LEVEL_KEYS))

