"""
review_risk_features.py

Purpose
-------
Research-stage review and validation of the selected behavioural risk
features before construction of the risk-scoring layer.

This script does NOT:
- train a risk model
- assign attacker intent
- create ground-truth labels
- modify the selected features automatically

It DOES:
- validate selected risk features
- inspect feature quality
- inspect statistical/effect-size evidence
- inspect redundancy
- handle an intentionally empty redundancy CSV
- identify potential concerns
- generate a human-review report
- generate a machine-readable JSON report
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

CLUSTER_RESULTS_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "clustering"
    / "results"
)

RISK_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "risk"
)

SELECTED_FEATURES_FILE = (
    RISK_DIR / "selected_risk_features.csv"
)

CANDIDATE_FEATURES_FILE = (
    RISK_DIR / "risk_feature_candidates.csv"
)

RISK_REDUNDANCY_FILE = (
    RISK_DIR / "risk_feature_redundancy.csv"
)

CLUSTER_STATISTICS_FILE = (
    CLUSTER_RESULTS_DIR / "cluster_statistical_tests.csv"
)

CLUSTER_EFFECTS_FILE = (
    CLUSTER_RESULTS_DIR / "cluster_effect_sizes.csv"
)

CLUSTER_FEATURE_EFFECTS_FILE = (
    CLUSTER_RESULTS_DIR / "cluster_feature_effects.csv"
)

CLUSTER_PROFILES_FILE = (
    CLUSTER_RESULTS_DIR / "cluster_profiles.csv"
)

OUTPUT_REPORT_FILE = (
    RISK_DIR / "risk_feature_review_report.json"
)

OUTPUT_REVIEW_FILE = (
    RISK_DIR / "risk_feature_review.csv"
)

OUTPUT_REDUNDANCY_FILE = (
    RISK_DIR / "risk_feature_redundancy_review.csv"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

SIGNIFICANCE_THRESHOLD = 0.05

# These are review thresholds, NOT attacker-risk thresholds.
STRONG_EFFECT_THRESHOLD = 0.80
MODERATE_EFFECT_THRESHOLD = 0.50
SMALL_EFFECT_THRESHOLD = 0.20

HIGH_ZERO_RATE = 0.80
HIGH_MISSING_RATE = 0.05

# Redundancy threshold used by the previous pipeline.
REDUNDANCY_THRESHOLD = 0.90


# ============================================================================
# OUTPUT HELPERS
# ============================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_section(title: str) -> None:
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


def ensure_output_directory() -> None:
    RISK_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# SAFE VALUE HELPERS
# ============================================================================

def safe_float(value: Any, default: float | None = None) -> float | None:
    """
    Convert a value to a finite float.

    NaN and infinity become default.
    """
    try:
        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except (TypeError, ValueError):
        return default


def json_safe(value: Any) -> Any:
    """
    Recursively convert pandas/numpy values into JSON-safe values.

    NaN and +/-inf are converted to None.
    """

    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

        return value

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(v)
            for v in value
        ]

    if isinstance(value, pd.Series):
        return json_safe(value.to_dict())

    if isinstance(value, pd.DataFrame):
        return json_safe(
            value.to_dict(orient="records")
        )

    if pd.isna(value):
        return None

    return value


# ============================================================================
# CSV LOADING
# ============================================================================

def load_csv(
    path: Path,
    description: str,
    allow_empty: bool = False,
) -> pd.DataFrame:

    print_header(f"LOADING {description.upper()}")

    print(f"Loading:")
    print(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Required file does not exist:\n{path}"
        )

    file_size = path.stat().st_size

    # Empty files are important in this project because the redundancy
    # selection stage may legitimately produce zero redundancy pairs.
    if file_size <= 2:

        if allow_empty:

            print()
            print(
                f"WARNING: {description} is empty."
            )

            print(
                "No records were generated."
            )

            print(
                "This is treated as an empty result, "
                "not as a fatal error."
            )

            return pd.DataFrame()

        raise ValueError(
            f"{description} exists but contains no data:\n{path}"
        )

    try:
        df = pd.read_csv(path)

    except pd.errors.EmptyDataError:

        if allow_empty:

            print()
            print(
                f"WARNING: {description} contains no data."
            )

            print(
                "Continuing with an empty DataFrame."
            )

            return pd.DataFrame()

        raise

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df


# ============================================================================
# COLUMN NORMALIZATION
# ============================================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        return df.copy()

    result = df.copy()

    result.columns = [
        str(column).strip()
        for column in result.columns
    ]

    return result


# ============================================================================
# SELECTED FEATURE VALIDATION
# ============================================================================
def extract_selected_features(selected_df):
    """
    Extract selected risk feature names.

    selected_risk_features.csv is a session-level dataset generated by
    select_risk_features.py.

    Format:

        session_id
        <risk feature 1>
        <risk feature 2>
        ...
        <risk feature N>

    The session_id column is an identifier and is NOT treated as a
    risk feature.
    """

    if selected_df is None:
        raise ValueError(
            "Selected risk feature dataframe is None."
        )

    if selected_df.empty:
        raise ValueError(
            "selected_risk_features.csv is empty."
        )

    columns = list(selected_df.columns)

    # --------------------------------------------------------------
    # FORMAT 1:
    # Session-level selected feature dataset
    # --------------------------------------------------------------

    if "session_id" in columns:

        feature_columns = [
            column
            for column in columns
            if column != "session_id"
        ]

        if not feature_columns:
            raise ValueError(
                "selected_risk_features.csv contains session_id "
                "but no risk feature columns."
            )

        # Make sure all selected features are numeric.
        non_numeric = []

        for feature in feature_columns:

            if not pd.api.types.is_numeric_dtype(
                selected_df[feature]
            ):
                non_numeric.append(feature)

        if non_numeric:
            raise ValueError(
                "Selected risk features must be numeric.\n"
                f"Non-numeric columns: {non_numeric}"
            )

        return feature_columns

    # --------------------------------------------------------------
    # FORMAT 2:
    # Feature-ranking table
    #
    # Kept for compatibility with older versions of the pipeline.
    # --------------------------------------------------------------

    if "feature" in columns:

        features = (
            selected_df["feature"]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )

        features = list(
            dict.fromkeys(features)
        )

        if not features:
            raise ValueError(
                "The feature column exists but contains no "
                "valid feature names."
            )

        return features

    # --------------------------------------------------------------
    # Unsupported format
    # --------------------------------------------------------------

    raise ValueError(
        "Unable to determine selected risk features.\n\n"
        "Expected either:\n"
        "  1. A session-level dataset containing 'session_id' "
        "plus feature columns, or\n"
        "  2. A feature-ranking dataset containing 'feature'.\n\n"
        f"Actual columns: {columns}"
    )

# ============================================================================
# CANDIDATE FEATURE REVIEW
# ============================================================================

def build_candidate_lookup(
    candidate_df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:

    if candidate_df.empty:
        return {}

    if "feature" not in candidate_df.columns:
        return {}

    lookup: dict[str, dict[str, Any]] = {}

    for _, row in candidate_df.iterrows():

        feature = str(
            row.get("feature", "")
        ).strip()

        if not feature:
            continue

        lookup[feature] = {
            "p_value": safe_float(
                row.get("p_value")
            ),
            "significant": bool(
                row.get("significant", False)
            ),
            "effect_size": safe_float(
                row.get("effect_size")
            ),
            "absolute_effect": safe_float(
                row.get("absolute_effect")
            ),
            "cliffs_delta": safe_float(
                row.get("cliffs_delta")
            ),
            "missing_count": safe_float(
                row.get("missing_count"),
                0,
            ),
            "infinite_count": safe_float(
                row.get("infinite_count"),
                0,
            ),
            "unique_values": safe_float(
                row.get("unique_values"),
                0,
            ),
            "zero_count": safe_float(
                row.get("zero_count"),
                0,
            ),
            "zero_rate": safe_float(
                row.get("zero_rate")
            ),
            "quality_score": safe_float(
                row.get("quality_score")
            ),
            "effect_category": (
                str(row.get("effect_category", "unknown"))
                if pd.notna(row.get("effect_category"))
                else "unknown"
            ),
        }

    return lookup


# ============================================================================
# EFFECT INTERPRETATION
# ============================================================================

def classify_effect(
    absolute_effect: float | None,
) -> str:

    if absolute_effect is None:
        return "unknown"

    if absolute_effect >= STRONG_EFFECT_THRESHOLD:
        return "strong"

    if absolute_effect >= MODERATE_EFFECT_THRESHOLD:
        return "moderate"

    if absolute_effect >= SMALL_EFFECT_THRESHOLD:
        return "small"

    return "negligible"


# ============================================================================
# FEATURE REVIEW TABLE
# ============================================================================

def build_feature_review(
    selected_features: list[str],
    candidate_lookup: dict[str, dict[str, Any]],
) -> pd.DataFrame:

    records = []

    for rank, feature in enumerate(
        selected_features,
        start=1,
    ):

        data = candidate_lookup.get(
            feature,
            {},
        )

        p_value = data.get(
            "p_value"
        )

        effect_size = data.get(
            "effect_size"
        )

        absolute_effect = data.get(
            "absolute_effect"
        )

        zero_rate = data.get(
            "zero_rate"
        )

        missing_count = data.get(
            "missing_count",
            0,
        )

        infinite_count = data.get(
            "infinite_count",
            0,
        )

        quality_score = data.get(
            "quality_score"
        )

        concerns = []

        if p_value is not None:
            if p_value > SIGNIFICANCE_THRESHOLD:
                concerns.append(
                    "not_statistically_significant"
                )

        else:
            concerns.append(
                "missing_p_value"
            )

        if absolute_effect is None:
            concerns.append(
                "missing_effect_size"
            )

        if (
            zero_rate is not None
            and zero_rate >= HIGH_ZERO_RATE
        ):
            concerns.append(
                "high_zero_rate"
            )

        if missing_count and missing_count > 0:
            concerns.append(
                "missing_values"
            )

        if infinite_count and infinite_count > 0:
            concerns.append(
                "infinite_values"
            )

        records.append(
            {
                "rank": rank,
                "feature": feature,
                "p_value": p_value,
                "significant": (
                    data.get("significant")
                ),
                "effect_size": effect_size,
                "absolute_effect": absolute_effect,
                "cliffs_delta": (
                    data.get("cliffs_delta")
                ),
                "effect_category": classify_effect(
                    absolute_effect
                ),
                "candidate_effect_category": (
                    data.get(
                        "effect_category"
                    )
                ),
                "missing_count": missing_count,
                "infinite_count": infinite_count,
                "zero_rate": zero_rate,
                "unique_values": (
                    data.get("unique_values")
                ),
                "quality_score": quality_score,
                "review_concerns": (
                    ";".join(concerns)
                    if concerns
                    else ""
                ),
            }
        )

    return pd.DataFrame(records)


# ============================================================================
# REDUNDANCY REVIEW
# ============================================================================

def normalize_redundancy_columns(
    redundancy_df: pd.DataFrame,
) -> pd.DataFrame:

    if redundancy_df.empty:
        return pd.DataFrame(
            columns=[
                "feature_a",
                "feature_b",
                "correlation",
                "absolute_correlation",
                "selected_a",
                "selected_b",
                "both_selected",
            ]
        )

    df = redundancy_df.copy()

    # Accept several possible names from earlier versions.
    rename_map = {}

    if (
        "spearman_correlation"
        in df.columns
        and "correlation"
        not in df.columns
    ):
        rename_map[
            "spearman_correlation"
        ] = "correlation"

    if (
        "absolute_correlation"
        not in df.columns
    ):
        if "abs_correlation" in df.columns:
            rename_map[
                "abs_correlation"
            ] = "absolute_correlation"

    df = df.rename(
        columns=rename_map
    )

    return df


def analyze_redundancy(
    redundancy_df: pd.DataFrame,
    selected_features: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:

    selected_set = set(
        selected_features
    )

    normalized = normalize_redundancy_columns(
        redundancy_df
    )

    if normalized.empty:

        print_section(
            "REDUNDANCY ANALYSIS"
        )

        print(
            "No redundancy pairs were found."
        )

        print(
            "This is acceptable."
        )

        print(
            "The selected risk-feature set contains "
            "no recorded redundancy pairs."
        )

        result = {
            "status": "no_redundancy_detected",
            "pair_count": 0,
            "selected_pair_count": 0,
            "threshold": REDUNDANCY_THRESHOLD,
            "pairs": [],
        }

        empty = pd.DataFrame(
            columns=[
                "feature_a",
                "feature_b",
                "correlation",
                "absolute_correlation",
                "selected_a",
                "selected_b",
                "both_selected",
            ]
        )

        return empty, result

    required = [
        "feature_a",
        "feature_b",
    ]

    missing = [
        column
        for column in required
        if column not in normalized.columns
    ]

    if missing:

        print(
            "WARNING: Redundancy file does not contain "
            f"required columns: {missing}"
        )

        result = {
            "status": "unavailable",
            "pair_count": 0,
            "selected_pair_count": 0,
            "threshold": REDUNDANCY_THRESHOLD,
            "pairs": [],
        }

        return pd.DataFrame(), result

    if (
        "absolute_correlation"
        not in normalized.columns
    ):

        if "correlation" in normalized.columns:

            normalized[
                "absolute_correlation"
            ] = pd.to_numeric(
                normalized["correlation"],
                errors="coerce",
            ).abs()

        else:

            normalized[
                "absolute_correlation"
            ] = np.nan

    normalized[
        "selected_a"
    ] = normalized[
        "feature_a"
    ].isin(selected_set)

    normalized[
        "selected_b"
    ] = normalized[
        "feature_b"
    ].isin(selected_set)

    normalized[
        "both_selected"
    ] = (
        normalized["selected_a"]
        & normalized["selected_b"]
    )

    selected_pairs = normalized[
        normalized["both_selected"]
    ].copy()

    selected_pairs = selected_pairs.sort_values(
        "absolute_correlation",
        ascending=False,
        na_position="last",
    )

    pairs = []

    for _, row in selected_pairs.iterrows():

        pairs.append(
            {
                "feature_a": str(
                    row["feature_a"]
                ),
                "feature_b": str(
                    row["feature_b"]
                ),
                "correlation": safe_float(
                    row.get("correlation")
                ),
                "absolute_correlation": safe_float(
                    row.get(
                        "absolute_correlation"
                    )
                ),
            }
        )

    result = {
        "status": (
            "redundancy_detected"
            if len(pairs) > 0
            else "no_selected_feature_redundancy"
        ),
        "pair_count": int(
            len(normalized)
        ),
        "selected_pair_count": int(
            len(pairs)
        ),
        "threshold": REDUNDANCY_THRESHOLD,
        "pairs": pairs,
    }

    print_section(
        "REDUNDANCY ANALYSIS"
    )

    print(
        f"Total redundancy records: "
        f"{len(normalized)}"
    )

    print(
        f"Pairs involving two selected features: "
        f"{len(selected_pairs)}"
    )

    if selected_pairs.empty:

        print()
        print(
            "No redundancy exists among the "
            "selected risk features."
        )

    else:

        print()

        print(
            selected_pairs[
                [
                    "feature_a",
                    "feature_b",
                    "absolute_correlation",
                ]
            ].to_string(
                index=False
            )
        )

    return normalized, result


# ============================================================================
# STATISTICAL EVIDENCE
# ============================================================================

def summarize_statistical_evidence(
    selected_features: list[str],
    candidate_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:

    significant = []
    non_significant = []
    missing_evidence = []

    for feature in selected_features:

        data = candidate_lookup.get(
            feature
        )

        if not data:

            missing_evidence.append(
                feature
            )

            continue

        p_value = data.get(
            "p_value"
        )

        if p_value is None:

            missing_evidence.append(
                feature
            )

        elif p_value <= SIGNIFICANCE_THRESHOLD:

            significant.append(
                feature
            )

        else:

            non_significant.append(
                feature
            )

    return {
        "significance_threshold": (
            SIGNIFICANCE_THRESHOLD
        ),
        "selected_feature_count": len(
            selected_features
        ),
        "significant_count": len(
            significant
        ),
        "non_significant_count": len(
            non_significant
        ),
        "missing_evidence_count": len(
            missing_evidence
        ),
        "significant_features": significant,
        "non_significant_features": (
            non_significant
        ),
        "missing_evidence_features": (
            missing_evidence
        ),
    }


# ============================================================================
# EFFECT SIZE SUMMARY
# ============================================================================

def summarize_effect_sizes(
    feature_review_df: pd.DataFrame,
) -> dict[str, Any]:

    if feature_review_df.empty:

        return {
            "strong": [],
            "moderate": [],
            "small": [],
            "negligible": [],
            "unknown": [],
        }

    result = {}

    for category in [
        "strong",
        "moderate",
        "small",
        "negligible",
        "unknown",
    ]:

        subset = feature_review_df[
            feature_review_df[
                "effect_category"
            ] == category
        ]

        result[category] = (
            subset["feature"]
            .tolist()
        )

    return result


# ============================================================================
# DATA QUALITY SUMMARY
# ============================================================================

def summarize_quality(
    feature_review_df: pd.DataFrame,
) -> dict[str, Any]:

    if feature_review_df.empty:

        return {
            "quality_issues": [],
            "high_zero_rate_features": [],
            "missing_value_features": [],
            "infinite_value_features": [],
        }

    quality_issues = []
    high_zero = []
    missing = []
    infinite = []

    for _, row in feature_review_df.iterrows():

        feature = row[
            "feature"
        ]

        zero_rate = safe_float(
            row.get("zero_rate")
        )

        missing_count = safe_float(
            row.get("missing_count"),
            0,
        )

        infinite_count = safe_float(
            row.get("infinite_count"),
            0,
        )

        concerns = []

        if (
            zero_rate is not None
            and zero_rate >= HIGH_ZERO_RATE
        ):

            high_zero.append(
                feature
            )

            concerns.append(
                "high_zero_rate"
            )

        if missing_count > 0:

            missing.append(
                feature
            )

            concerns.append(
                "missing_values"
            )

        if infinite_count > 0:

            infinite.append(
                feature
            )

            concerns.append(
                "infinite_values"
            )

        if concerns:

            quality_issues.append(
                {
                    "feature": feature,
                    "issues": concerns,
                }
            )

    return {
        "quality_issues": quality_issues,
        "high_zero_rate_features": high_zero,
        "missing_value_features": missing,
        "infinite_value_features": infinite,
    }


# ============================================================================
# REVIEW FLAGS
# ============================================================================

def build_review_flags(
    feature_review_df: pd.DataFrame,
    redundancy_summary: dict[str, Any],
    statistical_summary: dict[str, Any],
) -> list[dict[str, Any]]:

    flags = []

    # ---------------------------------------------------------
    # Statistical evidence
    # ---------------------------------------------------------

    for feature in statistical_summary[
        "non_significant_features"
    ]:

        flags.append(
            {
                "severity": "review",
                "feature": feature,
                "issue": (
                    "Feature is not statistically "
                    "significant under the configured "
                    "threshold."
                ),
            }
        )

    # ---------------------------------------------------------
    # Missing evidence
    # ---------------------------------------------------------

    for feature in statistical_summary[
        "missing_evidence_features"
    ]:

        flags.append(
            {
                "severity": "warning",
                "feature": feature,
                "issue": (
                    "Statistical evidence is missing "
                    "for this selected feature."
                ),
            }
        )

    # ---------------------------------------------------------
    # Quality
    # ---------------------------------------------------------

    quality_summary = summarize_quality(
        feature_review_df
    )

    for item in quality_summary[
        "quality_issues"
    ]:

        flags.append(
            {
                "severity": "review",
                "feature": item[
                    "feature"
                ],
                "issue": ", ".join(
                    item["issues"]
                ),
            }
        )

    # ---------------------------------------------------------
    # Redundancy
    # ---------------------------------------------------------

    if (
        redundancy_summary[
            "selected_pair_count"
        ] > 0
    ):

        for pair in redundancy_summary[
            "pairs"
        ]:

            correlation = pair[
                "absolute_correlation"
            ]

            if (
                correlation is not None
                and correlation
                >= REDUNDANCY_THRESHOLD
            ):

                flags.append(
                    {
                        "severity": "important",
                        "feature": (
                            f"{pair['feature_a']} / "
                            f"{pair['feature_b']}"
                        ),
                        "issue": (
                            "Selected features have "
                            f"high redundancy "
                            f"(absolute correlation="
                            f"{correlation:.4f})."
                        ),
                    }
                )

    return flags


# ============================================================================
# CLUSTER CONTEXT
# ============================================================================

def load_optional_cluster_context() -> dict[str, Any]:

    context: dict[str, Any] = {}

    files_to_load = {
        "cluster_profiles": (
            CLUSTER_PROFILES_FILE
        ),
        "cluster_effects": (
            CLUSTER_EFFECTS_FILE
        ),
        "cluster_feature_effects": (
            CLUSTER_FEATURE_EFFECTS_FILE
        ),
        "cluster_statistics": (
            CLUSTER_STATISTICS_FILE
        ),
    }

    for name, path in files_to_load.items():

        if not path.exists():

            context[name] = {
                "status": "not_available"
            }

            continue

        try:

            if path.stat().st_size <= 2:

                context[name] = {
                    "status": "empty"
                }

                continue

            df = pd.read_csv(path)

            context[name] = {
                "status": "available",
                "rows": len(df),
                "columns": list(
                    df.columns
                ),
            }

        except Exception as exc:

            context[name] = {
                "status": "error",
                "error": str(exc),
            }

    return context


# ============================================================================
# HUMAN REVIEW TABLE
# ============================================================================

def print_feature_review(
    feature_review_df: pd.DataFrame,
) -> None:

    print_section(
        "SELECTED RISK FEATURE REVIEW"
    )

    if feature_review_df.empty:

        print(
            "No feature review records."
        )

        return

    display_columns = [
        "rank",
        "feature",
        "p_value",
        "absolute_effect",
        "effect_category",
        "zero_rate",
        "quality_score",
        "review_concerns",
    ]

    available = [
        column
        for column in display_columns
        if column in feature_review_df.columns
    ]

    print(
        feature_review_df[
            available
        ].to_string(
            index=False
        )
    )


# ============================================================================
# RESEARCH INTERPRETATION
# ============================================================================

def generate_research_conclusion(
    selected_features: list[str],
    feature_review_df: pd.DataFrame,
    redundancy_summary: dict[str, Any],
    statistical_summary: dict[str, Any],
    flags: list[dict[str, Any]],
) -> dict[str, Any]:

    strong_count = len(
        statistical_summary[
            "significant_features"
        ]
    )

    redundancy_count = (
        redundancy_summary[
            "selected_pair_count"
        ]
    )

    serious_flags = [
        flag
        for flag in flags
        if flag["severity"]
        in {
            "important",
            "warning",
        }
    ]

    if redundancy_count == 0:

        redundancy_statement = (
            "No redundancy pairs were identified "
            "among the selected risk features."
        )

    else:

        redundancy_statement = (
            f"{redundancy_count} redundancy pair(s) "
            "were identified among the selected "
            "risk features and require review."
        )

    return {
        "selected_feature_count": len(
            selected_features
        ),
        "statistically_significant_count": (
            strong_count
        ),
        "statistical_significance_statement": (
            "Most or all selected features have "
            "statistical evidence in the current "
            "candidate analysis."
            if strong_count
            == len(selected_features)
            else
            "Not every selected feature has "
            "statistical evidence meeting the "
            "configured threshold."
        ),
        "redundancy_statement": (
            redundancy_statement
        ),
        "review_flag_count": len(
            flags
        ),
        "important_warning_count": len(
            serious_flags
        ),
        "ready_for_risk_scoring_design": (
            len(
                serious_flags
            ) == 0
        ),
        "research_caution": (
            "Feature selection and statistical "
            "separation do not establish attacker "
            "intent or ground-truth attacker labels."
        ),
    }


# ============================================================================
# REPORT BUILDING
# ============================================================================

def build_report(
    selected_features: list[str],
    feature_review_df: pd.DataFrame,
    redundancy_summary: dict[str, Any],
    statistical_summary: dict[str, Any],
    effect_summary: dict[str, Any],
    quality_summary: dict[str, Any],
    flags: list[dict[str, Any]],
    cluster_context: dict[str, Any],
) -> dict[str, Any]:

    conclusion = generate_research_conclusion(
        selected_features,
        feature_review_df,
        redundancy_summary,
        statistical_summary,
        flags,
    )

    return {
        "report_type": (
            "research_risk_feature_review"
        ),
        "version": "1.0",
        "status": "complete",

        "configuration": {
            "significance_threshold": (
                SIGNIFICANCE_THRESHOLD
            ),
            "redundancy_threshold": (
                REDUNDANCY_THRESHOLD
            ),
            "strong_effect_threshold": (
                STRONG_EFFECT_THRESHOLD
            ),
            "moderate_effect_threshold": (
                MODERATE_EFFECT_THRESHOLD
            ),
            "small_effect_threshold": (
                SMALL_EFFECT_THRESHOLD
            ),
            "high_zero_rate": (
                HIGH_ZERO_RATE
            ),
        },

        "selected_features": selected_features,

        "feature_review": (
            feature_review_df.to_dict(
                orient="records"
            )
        ),

        "statistical_evidence": (
            statistical_summary
        ),

        "effect_size_summary": (
            effect_summary
        ),

        "quality_summary": (
            quality_summary
        ),

        "redundancy": (
            redundancy_summary
        ),

        "review_flags": flags,

        "cluster_context": cluster_context,

        "conclusion": conclusion,

        "research_warning": (
            "The selected features are behavioural "
            "risk-scoring candidates. They are not "
            "ground-truth attacker labels and should "
            "not be interpreted as proof of malicious "
            "intent."
        ),

        "next_stage": (
            "Review this report before constructing "
            "the risk-scoring layer. Risk scoring "
            "should be validated separately from "
            "unsupervised cluster interpretation."
        ),
    }


# ============================================================================
# SAVE REPORTS
# ============================================================================

def save_outputs(
    feature_review_df: pd.DataFrame,
    redundancy_df: pd.DataFrame,
    report: dict[str, Any],
) -> None:

    ensure_output_directory()

    feature_review_df.to_csv(
        OUTPUT_REVIEW_FILE,
        index=False,
    )

    redundancy_df.to_csv(
        OUTPUT_REDUNDANCY_FILE,
        index=False,
    )

    with open(
        OUTPUT_REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_safe(report),
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    print_section(
        "SAVING REVIEW OUTPUTS"
    )

    print(
        f"Saved:\n{OUTPUT_REVIEW_FILE}"
    )

    print(
        f"Saved:\n{OUTPUT_REDUNDANCY_FILE}"
    )

    print(
        f"Saved:\n{OUTPUT_REPORT_FILE}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print()
    print("=" * 70)
    print(
        "RISK FEATURE REVIEW AND VALIDATION"
    )
    print("=" * 70)

    ensure_output_directory()

    # ------------------------------------------------------------------------
    # 1. LOAD SELECTED FEATURES
    # ------------------------------------------------------------------------

    selected_df = load_csv(
        SELECTED_FEATURES_FILE,
        "selected risk features",
        allow_empty=False,
    )

    selected_df = normalize_columns(
        selected_df
    )

    selected_features = extract_selected_features(
        selected_df
    )

    print()
    print(
        f"Selected features: "
        f"{len(selected_features)}"
    )

    for index, feature in enumerate(
        selected_features,
        start=1,
    ):

        print(
            f"{index:2d}. {feature}"
        )

    # ------------------------------------------------------------------------
    # 2. LOAD CANDIDATE FEATURES
    # ------------------------------------------------------------------------

    candidate_df = load_csv(
        CANDIDATE_FEATURES_FILE,
        "risk feature candidates",
        allow_empty=False,
    )

    candidate_df = normalize_columns(
        candidate_df
    )

    candidate_lookup = build_candidate_lookup(
        candidate_df
    )

    # ------------------------------------------------------------------------
    # 3. BUILD FEATURE REVIEW
    # ------------------------------------------------------------------------

    print_header(
        "BUILDING FEATURE REVIEW"
    )

    feature_review_df = build_feature_review(
        selected_features,
        candidate_lookup,
    )

    print_feature_review(
        feature_review_df
    )

    # ------------------------------------------------------------------------
    # 4. LOAD REDUNDANCY
    # ------------------------------------------------------------------------

    redundancy_df = load_csv(
        RISK_REDUNDANCY_FILE,
        "feature redundancy results",
        allow_empty=True,
    )

    redundancy_df = normalize_columns(
        redundancy_df
    )

    # ------------------------------------------------------------------------
    # 5. ANALYZE REDUNDANCY
    # ------------------------------------------------------------------------

    (
        reviewed_redundancy_df,
        redundancy_summary,
    ) = analyze_redundancy(
        redundancy_df,
        selected_features,
    )

    # ------------------------------------------------------------------------
    # 6. STATISTICAL EVIDENCE
    # ------------------------------------------------------------------------

    print_header(
        "STATISTICAL EVIDENCE REVIEW"
    )

    statistical_summary = (
        summarize_statistical_evidence(
            selected_features,
            candidate_lookup,
        )
    )

    print(
        "Selected features:"
        f" {statistical_summary['selected_feature_count']}"
    )

    print(
        "Statistically significant:"
        f" {statistical_summary['significant_count']}"
    )

    print(
        "Not statistically significant:"
        f" {statistical_summary['non_significant_count']}"
    )

    print(
        "Missing statistical evidence:"
        f" {statistical_summary['missing_evidence_count']}"
    )

    # ------------------------------------------------------------------------
    # 7. EFFECT SIZE
    # ------------------------------------------------------------------------

    print_header(
        "EFFECT SIZE REVIEW"
    )

    effect_summary = summarize_effect_sizes(
        feature_review_df
    )

    for category in [
        "strong",
        "moderate",
        "small",
        "negligible",
        "unknown",
    ]:

        features = effect_summary[
            category
        ]

        print(
            f"{category.capitalize():12s}: "
            f"{len(features)}"
        )

        for feature in features:

            print(
                f"    - {feature}"
            )

    # ------------------------------------------------------------------------
    # 8. QUALITY
    # ------------------------------------------------------------------------

    print_header(
        "FEATURE QUALITY REVIEW"
    )

    quality_summary = summarize_quality(
        feature_review_df
    )

    print(
        "High-zero-rate features:"
        f" {len(quality_summary['high_zero_rate_features'])}"
    )

    print(
        "Features with missing values:"
        f" {len(quality_summary['missing_value_features'])}"
    )

    print(
        "Features with infinite values:"
        f" {len(quality_summary['infinite_value_features'])}"
    )

    if quality_summary[
        "high_zero_rate_features"
    ]:

        print()

        for feature in quality_summary[
            "high_zero_rate_features"
        ]:

            print(
                f"WARNING: high zero rate -> "
                f"{feature}"
            )

    # ------------------------------------------------------------------------
    # 9. REVIEW FLAGS
    # ------------------------------------------------------------------------

    print_header(
        "BUILDING REVIEW FLAGS"
    )

    flags = build_review_flags(
        feature_review_df,
        redundancy_summary,
        statistical_summary,
    )

    if not flags:

        print(
            "No blocking review flags detected."
        )

    else:

        print(
            f"Review flags: {len(flags)}"
        )

        for flag in flags:

            print(
                f"[{flag['severity'].upper()}] "
                f"{flag['feature']}: "
                f"{flag['issue']}"
            )

    # ------------------------------------------------------------------------
    # 10. OPTIONAL CLUSTER CONTEXT
    # ------------------------------------------------------------------------

    print_header(
        "LOADING CLUSTER CONTEXT"
    )

    cluster_context = (
        load_optional_cluster_context()
    )

    for name, info in cluster_context.items():

        print(
            f"{name}: "
            f"{info.get('status')}"
        )

    # ------------------------------------------------------------------------
    # 11. BUILD REPORT
    # ------------------------------------------------------------------------

    print_header(
        "BUILDING RESEARCH REVIEW REPORT"
    )

    report = build_report(
        selected_features,
        feature_review_df,
        redundancy_summary,
        statistical_summary,
        effect_summary,
        quality_summary,
        flags,
        cluster_context,
    )

    # ------------------------------------------------------------------------
    # 12. SAVE
    # ------------------------------------------------------------------------

    save_outputs(
        feature_review_df,
        reviewed_redundancy_df,
        report,
    )

    # ------------------------------------------------------------------------
    # 13. FINAL SUMMARY
    # ------------------------------------------------------------------------

    print_header(
        "RISK FEATURE REVIEW COMPLETE"
    )

    conclusion = report[
        "conclusion"
    ]

    print(
        f"Selected features: "
        f"{conclusion['selected_feature_count']}"
    )

    print(
        f"Statistically significant: "
        f"{conclusion['statistically_significant_count']}"
    )

    print(
        f"Redundancy pairs among selected features: "
        f"{redundancy_summary['selected_pair_count']}"
    )

    print(
        f"Review flags: "
        f"{conclusion['review_flag_count']}"
    )

    print()

    if conclusion[
        "ready_for_risk_scoring_design"
    ]:

        print(
            "STATUS:"
        )

        print(
            "The selected feature set has no "
            "blocking review issues detected by "
            "this automated validation."
        )

    else:

        print(
            "STATUS:"
        )

        print(
            "Manual review is required before "
            "risk-scoring design."
        )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Statistical separation does not establish "
        "attacker intent."
    )

    print(
        "The selected features are behavioural "
        "risk-scoring candidates, not ground-truth "
        "attacker labels."
    )

    print()

    print(
        "Generated:"
    )

    print(
        f"1. {OUTPUT_REVIEW_FILE}"
    )

    print(
        f"2. {OUTPUT_REDUNDANCY_FILE}"
    )

    print(
        f"3. {OUTPUT_REPORT_FILE}"
    )

    print()

    print(
        "DO NOT TRAIN THE FINAL RISK MODEL YET."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()