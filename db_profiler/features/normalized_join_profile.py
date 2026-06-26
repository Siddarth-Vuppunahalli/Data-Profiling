from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations
from typing import Any, Callable, Mapping

import pandas as pd
from pandas.api.types import CategoricalDtype, is_object_dtype, is_string_dtype


Normalizer = Callable[[str], str]
JOIN_NAME_TOKENS = ("id", "key", "code", "uuid", "number", "num", "identifier")


def profile_normalized_join_compatibility(
    tables: Mapping[str, pd.DataFrame],
    max_collision_rate: float = 0.02,
    min_match_rate: float = 0.95,
) -> list[dict[str, Any]]:
    candidates = []
    for left_table, right_table in combinations(sorted(tables), 2):
        left_frame = tables[left_table]
        right_frame = tables[right_table]
        for left_column, right_column in _candidate_column_pairs(left_frame, right_frame):
            profile = profile_join_pair_compatibility(
                left_table=left_table,
                left_column=left_column,
                left_values=left_frame[left_column],
                right_table=right_table,
                right_column=right_column,
                right_values=right_frame[right_column],
                max_collision_rate=max_collision_rate,
            )
            if not profile["normalization_rejected"] and profile["_best_match_rate"] >= min_match_rate:
                profile.pop("_best_match_rate")
                candidates.append(profile)
    return candidates


def profile_join_pair_compatibility(
    left_table: str,
    left_column: str,
    left_values: pd.Series,
    right_table: str,
    right_column: str,
    right_values: pd.Series,
    max_collision_rate: float,
) -> dict[str, Any]:
    left_strings = _non_null_strings(left_values)
    right_strings = _non_null_strings(right_values)
    common_affixes = _common_affixes(left_strings, right_strings)

    normalizers: dict[str, Normalizer] = {
        "raw": _normalize_raw,
        "trimmed": _normalize_trimmed,
        "case": _normalize_case,
        "punctuation": _normalize_punctuation,
        "prefix_suffix": lambda value: _normalize_prefix_suffix(value, common_affixes),
    }
    rates = {
        name: _match_rate(left_strings, right_strings, normalizer)
        for name, normalizer in normalizers.items()
    }
    best_normalization, best_rate = _best_normalization(rates)
    best_normalizer = normalizers[best_normalization]
    collision_rate = max(
        _collision_rate(left_strings, best_normalizer),
        _collision_rate(right_strings, best_normalizer),
    )
    rejected = best_normalization != "raw" and collision_rate > max_collision_rate
    left_to_right, right_to_left = _coverage(left_strings, right_strings, best_normalizer)

    if rejected:
        public_best_normalization = "none"
        normalization_uplift = 0.0
    else:
        public_best_normalization = best_normalization
        normalization_uplift = round(max(0.0, best_rate - rates["raw"]), 4)

    return {
        "profiling_level": "table",
        "left_table": left_table,
        "left_column": left_column,
        "right_table": right_table,
        "right_column": right_column,
        "candidate_filter_reasons": ["compatible_column_name", "text_like", "sufficient_cardinality"],
        "is_confirmed_relationship": False,
        "raw_match_rate": rates["raw"],
        "trimmed_match_rate": rates["trimmed"],
        "case_normalized_match_rate": rates["case"],
        "punctuation_normalized_match_rate": rates["punctuation"],
        "prefix_suffix_normalized_match_rate": rates["prefix_suffix"],
        "best_performing_normalization": public_best_normalization,
        "normalization_uplift": normalization_uplift,
        "collision_rate_after_normalization": collision_rate,
        "normalization_rejected": rejected,
        "rejection_reason": "collision_rate_exceeds_threshold" if rejected else None,
        "left_to_right_coverage": left_to_right,
        "right_to_left_coverage": right_to_left,
        "_best_match_rate": 0.0 if rejected else best_rate,
    }


def _candidate_column_pairs(left_frame: pd.DataFrame, right_frame: pd.DataFrame) -> list[tuple[str, str]]:
    pairs = []
    for left_column in left_frame.columns:
        for right_column in right_frame.columns:
            if not _compatible_column_names(left_column, right_column):
                continue
            if not _is_text_column(left_frame[left_column]) or not _is_text_column(right_frame[right_column]):
                continue
            if not _sufficient_cardinality(left_frame[left_column], right_frame[right_column]):
                continue
            pairs.append((left_column, right_column))
    return pairs


def _compatible_column_names(left_column: str, right_column: str) -> bool:
    left = _normalize_column_name(left_column)
    right = _normalize_column_name(right_column)
    if left != right:
        return False
    return any(token in left.split("_") for token in JOIN_NAME_TOKENS)


def _normalize_column_name(column: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", column.strip().casefold()).strip("_")
    return normalized


def _is_text_column(values: pd.Series) -> bool:
    dtype = values.dtype
    return is_string_dtype(dtype) or is_object_dtype(dtype) or isinstance(dtype, CategoricalDtype)


def _sufficient_cardinality(left_values: pd.Series, right_values: pd.Series) -> bool:
    return max(_cardinality_ratio(left_values), _cardinality_ratio(right_values)) >= 0.80


def _cardinality_ratio(values: pd.Series) -> float:
    non_null = [value for value in values if not pd.isna(value)]
    if not non_null:
        return 0.0
    return len({str(value) for value in non_null}) / len(non_null)


def _non_null_strings(values: pd.Series) -> list[str]:
    return [str(value) for value in values if not pd.isna(value)]


def _normalize_raw(value: str) -> str:
    return value


def _normalize_trimmed(value: str) -> str:
    return value.strip()


def _normalize_case(value: str) -> str:
    return value.strip().casefold()


def _normalize_punctuation(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().casefold())


def _normalize_prefix_suffix(value: str, affixes: set[str]) -> str:
    normalized = _normalize_case(value)
    for affix in sorted(affixes, key=len, reverse=True):
        if normalized.startswith(f"{affix}-"):
            return normalized[len(affix) + 1 :]
        if normalized.startswith(f"{affix}_"):
            return normalized[len(affix) + 1 :]
        if normalized.endswith(f"-{affix}"):
            return normalized[: -(len(affix) + 1)]
        if normalized.endswith(f"_{affix}"):
            return normalized[: -(len(affix) + 1)]
    return normalized


def _common_affixes(left_values: list[str], right_values: list[str]) -> set[str]:
    left_affixes = _affixes(left_values)
    right_affixes = _affixes(right_values)
    return left_affixes & right_affixes


def _affixes(values: list[str]) -> set[str]:
    affixes = set()
    for value in values:
        normalized = _normalize_case(value)
        for delimiter in ("-", "_"):
            parts = [part for part in normalized.split(delimiter) if part]
            if len(parts) >= 2:
                affixes.add(parts[0])
                affixes.add(parts[-1])
    return affixes


def _match_rate(left_values: list[str], right_values: list[str], normalizer: Normalizer) -> float:
    left_set = {normalizer(value) for value in left_values}
    right_set = {normalizer(value) for value in right_values}
    if not left_set or not right_set:
        return 0.0
    denominator = min(len(left_set), len(right_set))
    return round(len(left_set & right_set) / denominator, 4)


def _coverage(left_values: list[str], right_values: list[str], normalizer: Normalizer) -> tuple[float, float]:
    left_set = {normalizer(value) for value in left_values}
    right_set = {normalizer(value) for value in right_values}
    overlap = left_set & right_set
    left_to_right = round(len(overlap) / len(left_set), 4) if left_set else 0.0
    right_to_left = round(len(overlap) / len(right_set), 4) if right_set else 0.0
    return left_to_right, right_to_left


def _best_normalization(rates: dict[str, float]) -> tuple[str, float]:
    for name in ("raw", "trimmed", "case", "punctuation", "prefix_suffix"):
        if rates[name] == max(rates.values()):
            return name, rates[name]
    return "raw", rates["raw"]


def _collision_rate(values: list[str], normalizer: Normalizer) -> float:
    raw_values = sorted(set(values))
    if not raw_values:
        return 0.0

    normalized_to_raw: dict[str, set[str]] = defaultdict(set)
    for value in raw_values:
        normalized_to_raw[normalizer(value)].add(value)

    collision_groups = sum(1 for raw_group in normalized_to_raw.values() if len(raw_group) > 1)
    return round(collision_groups / len(raw_values), 4)

