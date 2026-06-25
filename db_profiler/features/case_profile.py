from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd
from pandas.api.types import CategoricalDtype, is_object_dtype, is_string_dtype


CASE_KEYS = ("uppercase", "lowercase", "title_case", "mixed_case")


def profile_table_case_conventions(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        column: profile_case_conventions(frame[column])
        for column in frame.columns
        if _is_text_column(frame[column]) and _column_has_letters(frame[column])
    }


def profile_case_conventions(values: pd.Series) -> dict[str, Any]:
    non_null_values = [_stringify(value) for value in values if not pd.isna(value)]
    classified_cases = [_classify_case(value) for value in non_null_values if _has_letter(value)]
    case_counts = Counter(classified_cases)
    classified_count = len(classified_cases)
    collisions = _case_collisions(non_null_values)

    return {
        "profiling_level": "column",
        "total_non_null_count": len(non_null_values),
        "classified_value_count": classified_count,
        "uppercase_percentage": _percentage(case_counts["uppercase"], classified_count),
        "lowercase_percentage": _percentage(case_counts["lowercase"], classified_count),
        "title_case_percentage": _percentage(case_counts["title_case"], classified_count),
        "mixed_case_percentage": _percentage(case_counts["mixed_case"], classified_count),
        "dominant_case": _dominant_case(case_counts, classified_count),
        "raw_distinct_count": len(set(non_null_values)),
        "case_normalized_distinct_count": len({value.casefold() for value in non_null_values}),
        "case_collision_count": len(collisions),
        "case_collision_examples": collisions[:10],
    }


def _is_text_column(values: pd.Series) -> bool:
    dtype = values.dtype
    return is_string_dtype(dtype) or is_object_dtype(dtype) or isinstance(dtype, CategoricalDtype)


def _stringify(value: Any) -> str:
    return str(value)


def _has_letter(value: str) -> bool:
    return any(character.isalpha() for character in value)


def _column_has_letters(values: pd.Series) -> bool:
    return any(_has_letter(_stringify(value)) for value in values if not pd.isna(value))


def _classify_case(value: str) -> str:
    letters = [character for character in value if character.isalpha()]
    if all(character.isupper() for character in letters):
        return "uppercase"
    if all(character.islower() for character in letters):
        return "lowercase"
    if value.istitle():
        return "title_case"
    return "mixed_case"


def _percentage(count: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(count / denominator, 4)


def _dominant_case(case_counts: Counter[str], classified_count: int) -> str:
    if classified_count == 0:
        return "unknown"
    return max(CASE_KEYS, key=lambda key: case_counts[key])


def _case_collisions(values: list[str]) -> list[dict[str, Any]]:
    normalized_to_variants: dict[str, list[str]] = {}
    for value in values:
        normalized = value.casefold()
        variants = normalized_to_variants.setdefault(normalized, [])
        if value not in variants:
            variants.append(value)

    return [
        {
            "case_normalized_value": normalized,
            "raw_variants": variants,
        }
        for normalized, variants in normalized_to_variants.items()
        if len(variants) > 1
    ]
