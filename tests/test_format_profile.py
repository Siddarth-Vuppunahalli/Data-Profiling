import pandas as pd

from db_profiler.config import ProfilingConfig
from db_profiler.features.format_profile import profile_format_patterns
from db_profiler.runner import build_profile


class StubYDataProfiler:
    version = "stub"

    def profile_dataframe(self, table_name: str, frame: pd.DataFrame) -> dict:
        return {"table": {"n": len(frame)}, "variables": {}, "alerts": []}


def test_profile_format_patterns_reports_whitespace_patterns_and_lengths() -> None:
    values = pd.Series(
        [
            " SKU-001 ",
            "SKU-002",
            "SKU-003",
            "SKU-004",
            "SKU-ABC",
            "LEGACY",
            None,
        ],
        name="item_code",
    )

    profile = profile_format_patterns(values, min_affix_frequency=0.5)

    assert profile["profiling_level"] == "column"
    assert profile["total_non_null_count"] == 6
    assert profile["leading_whitespace_percentage"] == 0.1667
    assert profile["trailing_whitespace_percentage"] == 0.1667
    assert profile["raw_distinct_count"] == 6
    assert profile["trimmed_distinct_count"] == 6
    assert profile["dominant_pattern_signature"] == "AAA-000"
    assert profile["pattern_coverage"] == 0.6667
    assert profile["pattern_outlier_count"] == 2
    assert profile["pattern_outlier_examples"] == ["SKU-ABC", "LEGACY"]
    assert profile["common_prefixes"] == [{"prefix": "SKU", "count": 5, "percentage": 0.8333}]
    assert profile["common_suffixes"] == []
    assert profile["value_length_distribution"] == {
        "min": 6,
        "max": 9,
        "mean": 7.1667,
        "buckets": {"6": 1, "7": 4, "9": 1},
    }


def test_profile_format_patterns_reports_trimmed_distinct_count_without_modifying_values() -> None:
    values = pd.Series(["ABC", " ABC ", "ABC  ", "XYZ"], name="code")
    original = values.copy(deep=True)

    profile = profile_format_patterns(values, min_affix_frequency=0.5)

    assert profile["raw_distinct_count"] == 4
    assert profile["trimmed_distinct_count"] == 2
    assert values.equals(original)


def test_profile_format_patterns_omits_affixes_below_frequency_threshold() -> None:
    values = pd.Series(["AA-001", "BB-002", "CC-003"], name="code")

    profile = profile_format_patterns(values, min_affix_frequency=0.75)

    assert profile["common_prefixes"] == []
    assert profile["common_suffixes"] == []


def test_build_profile_includes_format_profiles_for_text_columns() -> None:
    config = ProfilingConfig()
    config.thresholds.min_affix_frequency = 0.5
    frame = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "item_code": [" SKU-001 ", "SKU-002", "LEGACY"],
            "updated_at": ["2026-01-01", "2026-01-02", "2026-01-03"],
        }
    )

    document = build_profile(
        tables={"orders": frame},
        config=config,
        ydata_profiler=StubYDataProfiler(),
    )

    format_profiles = document["tables"]["orders"]["custom_profiles"]["format_profile"]
    assert set(format_profiles) == {"item_code"}
    assert format_profiles["item_code"]["leading_whitespace_percentage"] == 0.3333
    assert format_profiles["item_code"]["dominant_pattern_signature"] == "AAA-000"
