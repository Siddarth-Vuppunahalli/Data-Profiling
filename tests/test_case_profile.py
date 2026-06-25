import pandas as pd

from db_profiler.config import ProfilingConfig
from db_profiler.features.case_profile import profile_case_conventions
from db_profiler.runner import build_profile


class StubYDataProfiler:
    version = "stub"

    def profile_dataframe(self, table_name: str, frame: pd.DataFrame) -> dict:
        return {"table": {"n": len(frame)}, "variables": {}, "alerts": []}


def test_profile_case_conventions_reports_distribution_and_collisions() -> None:
    values = pd.Series(
        [
            "ACTIVE",
            "Active",
            "active",
            "PENDING",
            "12345",
            "",
            None,
        ],
        name="status",
    )

    profile = profile_case_conventions(values)

    assert profile["profiling_level"] == "column"
    assert profile["total_non_null_count"] == 6
    assert profile["classified_value_count"] == 4
    assert profile["uppercase_percentage"] == 0.5
    assert profile["lowercase_percentage"] == 0.25
    assert profile["title_case_percentage"] == 0.25
    assert profile["mixed_case_percentage"] == 0.0
    assert profile["dominant_case"] == "uppercase"
    assert profile["raw_distinct_count"] == 6
    assert profile["case_normalized_distinct_count"] == 4
    assert profile["case_collision_count"] == 1
    assert profile["case_collision_examples"] == [
        {
            "case_normalized_value": "active",
            "raw_variants": ["ACTIVE", "Active", "active"],
        }
    ]


def test_profile_case_conventions_excludes_values_without_letters_from_classification() -> None:
    values = pd.Series(["123", " - ", "", None], name="code")

    profile = profile_case_conventions(values)

    assert profile["total_non_null_count"] == 3
    assert profile["classified_value_count"] == 0
    assert profile["uppercase_percentage"] == 0.0
    assert profile["lowercase_percentage"] == 0.0
    assert profile["title_case_percentage"] == 0.0
    assert profile["mixed_case_percentage"] == 0.0
    assert profile["dominant_case"] == "unknown"
    assert profile["case_collision_count"] == 0
    assert profile["case_collision_examples"] == []


def test_profile_case_conventions_does_not_mutate_input_values() -> None:
    values = pd.Series([" ACTIVE ", "active"], name="status")
    original = values.copy(deep=True)

    profile_case_conventions(values)

    assert values.equals(original)


def test_build_profile_includes_case_profiles_for_text_columns() -> None:
    frame = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "status": ["ACTIVE", "Active", "active"],
            "notes": ["VIP", None, "vip"],
            "updated_at": ["2026-01-01", "2026-01-02", "2026-01-03"],
        }
    )

    document = build_profile(
        tables={"customers": frame},
        config=ProfilingConfig(),
        ydata_profiler=StubYDataProfiler(),
    )

    case_profiles = document["tables"]["customers"]["custom_profiles"]["case_profile"]
    assert set(case_profiles) == {"status", "notes"}
    assert case_profiles["status"]["case_collision_count"] == 1
    assert case_profiles["notes"]["classified_value_count"] == 2
