import pandas as pd

from db_profiler.config import ProfilingConfig
from db_profiler.features.normalized_join_profile import (
    profile_join_pair_compatibility,
    profile_normalized_join_compatibility,
)
from db_profiler.runner import build_profile


class StubYDataProfiler:
    version = "stub"

    def profile_dataframe(self, table_name: str, frame: pd.DataFrame) -> dict:
        return {"table": {"n": len(frame)}, "variables": {}, "alerts": []}


def test_profile_join_pair_reports_safe_normalization_match_rates() -> None:
    parent = pd.Series(["ABC-001", "ABC-002", "ABC-003"], name="customer_code")
    child = pd.Series(["abc001", " abc002 ", "ABC.003", "MISSING"], name="customer_code")

    profile = profile_join_pair_compatibility(
        left_table="customers",
        left_column="customer_code",
        left_values=parent,
        right_table="orders",
        right_column="customer_code",
        right_values=child,
        max_collision_rate=0.10,
    )

    assert profile["profiling_level"] == "table"
    assert profile["left_table"] == "customers"
    assert profile["right_table"] == "orders"
    assert profile["left_column"] == "customer_code"
    assert profile["right_column"] == "customer_code"
    assert profile["is_confirmed_relationship"] is False
    assert profile["raw_match_rate"] == 0.0
    assert profile["trimmed_match_rate"] == 0.0
    assert profile["case_normalized_match_rate"] == 0.0
    assert profile["punctuation_normalized_match_rate"] == 1.0
    assert profile["prefix_suffix_normalized_match_rate"] == 0.0
    assert profile["best_performing_normalization"] == "punctuation"
    assert profile["normalization_uplift"] == 1.0
    assert profile["collision_rate_after_normalization"] == 0.0
    assert profile["normalization_rejected"] is False
    assert profile["left_to_right_coverage"] == 1.0
    assert profile["right_to_left_coverage"] == 0.75


def test_profile_join_pair_rejects_normalization_with_excessive_collisions() -> None:
    left = pd.Series(["AB-1", "A-B1", "CD-2"], name="code")
    right = pd.Series(["AB1", "CD2"], name="code")

    profile = profile_join_pair_compatibility(
        left_table="left_table",
        left_column="code",
        left_values=left,
        right_table="right_table",
        right_column="code",
        right_values=right,
        max_collision_rate=0.10,
    )

    assert profile["punctuation_normalized_match_rate"] == 1.0
    assert profile["collision_rate_after_normalization"] == 0.3333
    assert profile["normalization_rejected"] is True
    assert profile["rejection_reason"] == "collision_rate_exceeds_threshold"
    assert profile["best_performing_normalization"] == "none"


def test_profile_join_pair_does_not_mutate_input_values() -> None:
    left = pd.Series([" ABC-001 "], name="code")
    right = pd.Series(["abc001"], name="code")
    original_left = left.copy(deep=True)
    original_right = right.copy(deep=True)

    profile_join_pair_compatibility(
        left_table="customers",
        left_column="code",
        left_values=left,
        right_table="orders",
        right_column="code",
        right_values=right,
        max_collision_rate=0.10,
    )

    assert left.equals(original_left)
    assert right.equals(original_right)


def test_profile_normalized_join_compatibility_filters_candidate_columns() -> None:
    tables = {
        "customers": pd.DataFrame(
            {
                "customer_code": ["ABC-001", "ABC-002", "ABC-003"],
                "status": ["ACTIVE", "ACTIVE", "INACTIVE"],
            }
        ),
        "orders": pd.DataFrame(
            {
                "customer_code": ["abc001", "abc002", "MISSING"],
                "status": ["ACTIVE", "PENDING", "PENDING"],
            }
        ),
    }

    candidates = profile_normalized_join_compatibility(
        tables,
        max_collision_rate=0.10,
        min_match_rate=0.50,
    )

    assert len(candidates) == 1
    assert candidates[0]["left_column"] == "customer_code"
    assert candidates[0]["right_column"] == "customer_code"
    assert candidates[0]["candidate_filter_reasons"] == ["compatible_column_name", "text_like", "sufficient_cardinality"]


def test_build_profile_wires_normalized_join_candidates_into_relationships() -> None:
    tables = {
        "customers": pd.DataFrame({"customer_code": ["ABC-001", "ABC-002"]}),
        "orders": pd.DataFrame({"customer_code": ["abc001", "abc002"]}),
    }

    document = build_profile(
        tables=tables,
        config=ProfilingConfig(),
        ydata_profiler=StubYDataProfiler(),
    )

    candidates = document["relationships"]["normalized_join_candidates"]
    assert len(candidates) == 1
    assert candidates[0]["best_performing_normalization"] == "punctuation"
    assert candidates[0]["is_confirmed_relationship"] is False
