import json

import pandas as pd

from db_profiler.ydata_profile import YDataProfiler, compact_ydata_profile


class FakeProfileReport:
    def __init__(self, frame: pd.DataFrame, title: str, explorative: bool) -> None:
        assert title == "YData Profile: customers"
        assert explorative is True
        self.frame = frame

    def to_json(self) -> str:
        return json.dumps(
            {
                "table": {
                    "n": len(self.frame),
                    "n_var": len(self.frame.columns),
                    "n_cells_missing": 1,
                    "ignored_large_blob": "skip",
                },
                "variables": {
                    "customer_id": {
                        "type": "Numeric",
                        "n_distinct": 3,
                        "p_distinct": 1.0,
                        "n_missing": 0,
                        "ignored": "skip",
                    },
                    "name": {
                        "type": "Text",
                        "n_distinct": 3,
                        "p_distinct": 1.0,
                        "n_missing": 1,
                    },
                },
                "alerts": [{"type": "missing"}],
            }
        )


def test_compact_ydata_profile_keeps_useful_table_and_variable_metrics() -> None:
    compact = compact_ydata_profile(
        {
            "table": {"n": 3, "n_var": 2, "ignored": "skip"},
            "variables": {"name": {"type": "Text", "n_missing": 0, "ignored": "skip"}},
            "alerts": [],
        }
    )

    assert compact["table"] == {"n": 3, "n_var": 2}
    assert compact["variables"]["name"] == {"type": "Text", "n_missing": 0}
    assert compact["alerts"] == []


def test_ydata_profiler_uses_profile_report_and_returns_compact_json() -> None:
    frame = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "name": ["Ada", None, "Grace"],
        }
    )
    profiler = YDataProfiler(profile_report_cls=FakeProfileReport)

    result = profiler.profile_dataframe("customers", frame)

    assert result["table"]["n"] == 3
    assert result["table"]["n_var"] == 2
    assert result["table"]["n_cells_missing"] == 1
    assert result["variables"]["customer_id"]["type"] == "Numeric"
    assert result["variables"]["name"]["n_missing"] == 1
    assert result["alerts"] == [{"type": "missing"}]
    assert "raw" not in result

