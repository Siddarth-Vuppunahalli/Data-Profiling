from __future__ import annotations

import json
from typing import Any, Callable

import pandas as pd


class YDataProfiler:
    def __init__(
        self,
        profile_report_cls: Callable[..., Any] | None = None,
        include_raw: bool = False,
        explorative: bool = True,
    ) -> None:
        self.include_raw = include_raw
        self.explorative = explorative

        if profile_report_cls is None:
            try:
                from ydata_profiling import ProfileReport
                import ydata_profiling
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "ydata-profiling is required for profiling. Install project dependencies first."
                ) from exc

            self._profile_report_cls = ProfileReport
            self.version = getattr(ydata_profiling, "__version__", "unknown")
        else:
            self._profile_report_cls = profile_report_cls
            self.version = "test-double"

    def profile_dataframe(self, table_name: str, frame: pd.DataFrame) -> dict[str, Any]:
        report = self._profile_report_cls(
            frame,
            title=f"YData Profile: {table_name}",
            explorative=self.explorative,
        )
        raw = json.loads(report.to_json())
        compact = compact_ydata_profile(raw)
        if self.include_raw:
            compact["raw"] = raw
        return compact


def compact_ydata_profile(raw: dict[str, Any]) -> dict[str, Any]:
    table = raw.get("table", {})
    variables = raw.get("variables", {})

    return {
        "table": _select_keys(
            table,
            (
                "n",
                "n_var",
                "memory_size",
                "record_size",
                "n_cells_missing",
                "p_cells_missing",
                "n_duplicates",
                "p_duplicates",
            ),
        ),
        "variables": {
            name: _select_keys(
                stats,
                (
                    "type",
                    "n_distinct",
                    "p_distinct",
                    "n_missing",
                    "p_missing",
                    "n_unique",
                    "p_unique",
                    "mean",
                    "min",
                    "max",
                    "memory_size",
                ),
            )
            for name, stats in variables.items()
        },
        "alerts": raw.get("alerts", []),
    }


def _select_keys(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: source[key] for key in keys if key in source}

