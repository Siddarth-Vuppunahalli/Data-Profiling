from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

import pandas as pd
from pandas.api.types import CategoricalDtype, is_object_dtype, is_string_dtype


DELIMITERS = ("-", "_", "/", " ", ".")


def profile_table_format_patterns(
    frame: pd.DataFrame,
    min_affix_frequency: float = 0.10,
) -> dict[str, dict[str, Any]]:
    return {
        column: profile_format_patterns(frame[column], min_affix_frequency=min_affix_frequency)
        for column in frame.columns
        if _is_text_column(frame[column]) and _column_has_letters(frame[column])
    }


def profile_format_patterns(
    values: pd.Series,
    min_affix_frequency: float = 0.10,
) -> dict[str, Any]:
    non_null_values = [_stringify(value) for value in values if not pd.isna(value)]
    trimmed_values = [value.strip() for value in non_null_values]
    pattern_counts = Counter(_pattern_signature(value) for value in trimmed_values)
    dominant_pattern, dominant_count = _dominant_pattern(pattern_counts)
    outlier_values = [value for value in trimmed_values if _pattern_signature(value) != dominant_pattern]
    lengths = [len(value) for value in non_null_values]

    return {
        "profiling_level": "column",
        "total_non_null_count": len(non_null_values),
        "leading_whitespace_percentage": _percentage(
            sum(1 for value in non_null_values if value != value.lstrip()),
            len(non_null_values),
        ),
        "trailing_whitespace_percentage": _percentage(
            sum(1 for value in non_null_values if value != value.rstrip()),
            len(non_null_values),
        ),
        "raw_distinct_count": len(set(non_null_values)),
        "trimmed_distinct_count": len(set(trimmed_values)),
        "dominant_pattern_signature": dominant_pattern,
        "pattern_coverage": _percentage(dominant_count, len(non_null_values)),
        "pattern_outlier_count": len(outlier_values),
        "pattern_outlier_examples": _unique_in_order(outlier_values)[:10],
        "common_prefixes": _common_affixes(trimmed_values, side="prefix", min_frequency=min_affix_frequency),
        "common_suffixes": _common_affixes(trimmed_values, side="suffix", min_frequency=min_affix_frequency),
        "value_length_distribution": _length_distribution(lengths),
    }


def _is_text_column(values: pd.Series) -> bool:
    dtype = values.dtype
    return is_string_dtype(dtype) or is_object_dtype(dtype) or isinstance(dtype, CategoricalDtype)


def _column_has_letters(values: pd.Series) -> bool:
    return any(any(character.isalpha() for character in _stringify(value)) for value in values if not pd.isna(value))


def _stringify(value: Any) -> str:
    return str(value)


def _pattern_signature(value: str) -> str:
    signature = []
    for character in value:
        if character.isalpha():
            signature.append("A")
        elif character.isdigit():
            signature.append("0")
        else:
            signature.append(character)
    return "".join(signature)


def _dominant_pattern(pattern_counts: Counter[str]) -> tuple[str, int]:
    if not pattern_counts:
        return "unknown", 0
    return pattern_counts.most_common(1)[0]


def _common_affixes(values: list[str], side: str, min_frequency: float) -> list[dict[str, Any]]:
    total = len(values)
    if total == 0:
        return []

    counts: Counter[str] = Counter()
    for value in values:
        affix = _extract_affix(value, side)
        if affix is not None:
            counts[affix] += 1

    minimum_count = max(1, total * min_frequency)
    return [
        {
            side: affix,
            "count": count,
            "percentage": _percentage(count, total),
        }
        for affix, count in counts.most_common()
        if count >= minimum_count
    ]


def _extract_affix(value: str, side: str) -> str | None:
    for delimiter in DELIMITERS:
        if delimiter not in value:
            continue
        parts = [part for part in value.split(delimiter) if part]
        if len(parts) < 2:
            continue
        affix = parts[0] if side == "prefix" else parts[-1]
        if len(affix) >= 2 and any(character.isalpha() for character in affix):
            return affix
    return None


def _length_distribution(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0, "buckets": {}}

    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(mean(lengths), 4),
        "buckets": {str(length): count for length, count in sorted(Counter(lengths).items())},
    }


def _percentage(count: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(count / denominator, 4)


def _unique_in_order(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique

