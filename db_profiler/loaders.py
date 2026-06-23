from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv_tables(table_specs: list[str], sample_rows: int | None = None) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for spec in table_specs:
        table_name, csv_path = _parse_table_spec(spec)
        frame = pd.read_csv(csv_path)
        if sample_rows is not None:
            frame = frame.head(sample_rows)
        tables[table_name] = frame
    return tables


def _parse_table_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected TABLE=PATH for --table-csv, got: {spec}")

    table_name, path_text = spec.split("=", 1)
    table_name = table_name.strip()
    if not table_name:
        raise ValueError(f"Table name is required for --table-csv, got: {spec}")

    path = Path(path_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"CSV fixture does not exist: {path}")
    return table_name, path

