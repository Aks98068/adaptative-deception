"""
select_risk_features.py

Redundancy-aware risk feature selection for the
Adaptive Deception research pipeline.

Purpose
-------
Select a defensible subset of behavioural features for the
future risk-scoring layer.

This script DOES NOT train a risk model.

Inputs
------
1. research_features.csv
2. cluster_statistical_tests.csv
3. cluster_effect_sizes.csv
4. cluster_feature_redundancy.csv
5. cluster_validation_report.json

Outputs
-------
data/processed/risk/
    risk_feature_candidates.csv
    selected_risk_features.csv
    risk_feature_redundancy.csv
    risk_feature_selection_report.json
    risk_feature_configuration.json

Important
---------
Statistical significance does not establish attacker intent.

Cluster names remain behavioural descriptions rather than
ground-truth attacker labels.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ======================================================================
# PATH CONFIGURATION
# ======================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"

RESEARCH_DATASET = DATA_DIR / "research_features.csv"

CLUSTER_RESULTS_DIR = (
    DATA_DIR
    / "clustering"
    / "results"
)

STATISTICAL_TESTS = (
    CLUSTER_RESULTS_DIR
    / "cluster_statistical_tests.csv"
)

EFFECT_SIZES = (
    CLUSTER_RESULTS_DIR
    / "cluster_effect_sizes.csv"
)

REDUNDANCY_FILE = (
    CLUSTER_RESULTS_DIR
    / "cluster_feature_redundancy.csv"
)

VALIDATION_REPORT = (
    CLUSTER_RESULTS_DIR
    / "cluster_validation_report.json"
)

RISK_DIR = DATA_DIR / "risk"

RISK_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================
# RESEARCH FEATURE DEFINITIONS
# ======================================================================

RESEARCH_FEATURES = [
    "duration_sec",
    "average_event_interval",
    "event_interval_variance",
    "event_count",
    "unique_event_types",
    "unique_event_transitions",
    "failed_login_count",
    "successful_login_count",
    "failed_auth_ratio",
    "unique_usernames",
    "unique_hassh",
    "num_commands",
    "command_entropy",
    "num_file_events",
    "time_to_failed_login_sec",
    "time_to_successful_login_sec",
    "time_to_first_command_sec",
    "time_to_first_file_event_sec",
    "session_stage_encoded",
]


# ======================================================================
# FEATURES TO EXCLUDE FROM RISK SCORING
# ======================================================================

# These features are either:
#
# - highly redundant with another feature
# - derived from another feature
# - unsuitable for direct risk interpretation
# - weakly informative in this dataset
#
# The final decision is still determined by the statistical evidence
# and redundancy analysis below.

PREFERRED_FEATURES = [
    "failed_auth_ratio",
    "failed_login_count",
    "successful_login_count",
    "duration_sec",
    "average_event_interval",
    "event_interval_variance",
    "unique_event_types",
    "unique_event_transitions",
    "session_stage_encoded",
    "num_commands",
    "num_file_events",
    "time_to_failed_login_sec",
    "time_to_successful_login_sec",
    "time_to_first_command_sec",
    "time_to_first_file_event_sec",
    "unique_usernames",
    "unique_hassh",
    "command_entropy",
    "event_count",
]


# ======================================================================
# CONFIGURATION
# ======================================================================

SIGNIFICANCE_LEVEL = 0.05

STRONG_EFFECT_THRESHOLD = 0.80

MODERATE_EFFECT_THRESHOLD = 0.50

MIN_EFFECT_THRESHOLD = 0.10

REDUNDANCY_THRESHOLD = 0.90

VERY_HIGH_REDUNDANCY_THRESHOLD = 0.95

MAX_SELECTED_FEATURES = 12


# ======================================================================
# JSON SANITIZATION
# ======================================================================

def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert NaN and infinite values into JSON-safe values.

    NaN / +inf / -inf are converted to None so that:

        allow_nan=False

    can safely be used with json.dump().
    """

    if isinstance(obj, dict):
        return {
            str(key): sanitize_for_json(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            sanitize_for_json(value)
            for value in obj
        ]

    if isinstance(obj, tuple):
        return [
            sanitize_for_json(value)
            for value in obj
        ]

    if isinstance(obj, np.ndarray):
        return sanitize_for_json(
            obj.tolist()
        )

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        value = float(obj)

        if not math.isfinite(value):
            return None

        return value

    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None

        return obj

    if pd.isna(obj):
        return None

    return obj


# ======================================================================
# SAFE NUMERIC CONVERSION
# ======================================================================

def safe_numeric(
    series: pd.Series,
) -> pd.Series:
    """
    Convert a pandas series to numeric safely.
    Invalid values become NaN.
    """

    return pd.to_numeric(
        series,
        errors="coerce"
    )


# ======================================================================
# COLUMN UTILITIES
# ======================================================================

def find_column(
    df: pd.DataFrame,
    candidates: List[str],
) -> Optional[str]:
    """
    Return the first matching column from candidates.
    Matching is case-insensitive.
    """

    mapping = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        key = candidate.lower()

        if key in mapping:
            return mapping[key]

    return None


def ensure_column(
    df: pd.DataFrame,
    column: str,
    default: Any = np.nan,
) -> None:
    """
    Add a missing column with a default value.
    """

    if column not in df.columns:
        df[column] = default


# ======================================================================
# LOAD DATA
# ======================================================================

def load_csv(
    path: Path,
    required: bool = True,
) -> pd.DataFrame:
    """
    Load a CSV file.
    """

    if not path.exists():

        if required:
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

        return pd.DataFrame()

    print(f"Loading:")
    print(path)

    df = pd.read_csv(path)

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    return df


def load_json(
    path: Path,
) -> Dict[str, Any]:

    if not path.exists():
        return {}

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ======================================================================
# VALIDATE RESEARCH DATASET
# ======================================================================

def validate_research_dataset(
    df: pd.DataFrame,
) -> None:

    print()
    print("=" * 70)
    print("VALIDATING RESEARCH DATASET")
    print("=" * 70)

    missing = [
        feature
        for feature in RESEARCH_FEATURES
        if feature not in df.columns
    ]

    if missing:

        print("Missing research features:")

        for feature in missing:
            print(f" - {feature}")

        raise ValueError(
            "Required research features are missing."
        )

    print(
        f"Research features available: "
        f"{len(RESEARCH_FEATURES)}"
    )

    print("Validation successful.")


# ======================================================================
# NORMALIZE STATISTICAL TEST DATA
# ======================================================================

def normalize_statistical_tests(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("NORMALIZING STATISTICAL TEST RESULTS")
    print("=" * 70)

    if df.empty:
        raise ValueError(
            "cluster_statistical_tests.csv is empty."
        )

    feature_column = find_column(
        df,
        [
            "feature",
            "feature_name",
            "variable",
        ]
    )

    if feature_column is None:
        raise ValueError(
            "Could not identify feature column in "
            "cluster_statistical_tests.csv"
        )

    if feature_column != "feature":
        df = df.rename(
            columns={
                feature_column: "feature"
            }
        )

    # Normalize possible p-value column names.

    p_column = find_column(
        df,
        [
            "p_value",
            "pvalue",
            "p",
            "p-value",
        ]
    )

    if p_column is not None:
        df["p_value"] = safe_numeric(
            df[p_column]
        )
    else:
        df["p_value"] = np.nan

    # Normalize significance.

    sig_column = find_column(
        df,
        [
            "significant",
            "is_significant",
            "statistically_significant",
        ]
    )

    if sig_column is not None:

        values = (
            df[sig_column]
            .astype(str)
            .str.lower()
        )

        df["significant"] = values.isin(
            [
                "true",
                "1",
                "yes",
            ]
        )

    else:

        df["significant"] = (
            df["p_value"]
            .notna()
            &
            (
                df["p_value"]
                <= SIGNIFICANCE_LEVEL
            )
        )

    return df[
        [
            "feature",
            "p_value",
            "significant",
        ]
    ].drop_duplicates(
        subset=["feature"]
    )


# ======================================================================
# NORMALIZE EFFECT SIZES
# ======================================================================

def normalize_effect_sizes(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("NORMALIZING EFFECT SIZE RESULTS")
    print("=" * 70)

    if df.empty:
        raise ValueError(
            "cluster_effect_sizes.csv is empty."
        )

    feature_column = find_column(
        df,
        [
            "feature",
            "feature_name",
            "variable",
        ]
    )

    if feature_column is None:
        raise ValueError(
            "Could not identify feature column in "
            "cluster_effect_sizes.csv"
        )

    if feature_column != "feature":

        df = df.rename(
            columns={
                feature_column: "feature"
            }
        )

    # ================================================================
    # EFFECT SIZE
    # ================================================================

    effect_column = find_column(
        df,
        [
            "cliffs_delta",
            "effect_size",
            "absolute_effect",
            "effect",
            "cohens_d",
            "hedges_g",
        ]
    )

    if effect_column is not None:

        df["effect_size"] = safe_numeric(
            df[effect_column]
        )

    else:

        df["effect_size"] = np.nan

    # ================================================================
    # ABSOLUTE EFFECT
    # ================================================================

    absolute_column = find_column(
        df,
        [
            "absolute_effect",
            "abs_effect",
            "absolute_effect_size",
        ]
    )

    if absolute_column is not None:

        df["absolute_effect"] = safe_numeric(
            df[absolute_column]
        )

    else:

        df["absolute_effect"] = (
            df["effect_size"]
            .abs()
        )

    # ================================================================
    # CLIFF'S DELTA
    # ================================================================

    cliffs_column = find_column(
        df,
        [
            "cliffs_delta",
            "cliff_delta",
            "cliffs",
        ]
    )

    if cliffs_column is not None:

        df["cliffs_delta"] = safe_numeric(
            df[cliffs_column]
        )

    else:

        # This is important because your previous
        # version expected cliffs_delta even though
        # the CSV did not contain it.

        df["cliffs_delta"] = np.nan

    return df[
        [
            "feature",
            "effect_size",
            "absolute_effect",
            "cliffs_delta",
        ]
    ].drop_duplicates(
        subset=["feature"]
    )


# ======================================================================
# NORMALIZE REDUNDANCY DATA
# ======================================================================

def normalize_redundancy(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("NORMALIZING REDUNDANCY RESULTS")
    print("=" * 70)

    if df.empty:
        print(
            "No redundancy file found or file is empty."
        )

        return pd.DataFrame(
            columns=[
                "feature_a",
                "feature_b",
                "absolute_correlation",
            ]
        )

    feature_a = find_column(
        df,
        [
            "feature_a",
            "feature1",
            "feature_1",
        ]
    )

    feature_b = find_column(
        df,
        [
            "feature_b",
            "feature2",
            "feature_2",
        ]
    )

    correlation = find_column(
        df,
        [
            "absolute_correlation",
            "absolute_corr",
            "spearman_correlation",
            "correlation",
        ]
    )

    if (
        feature_a is None
        or feature_b is None
    ):

        print(
            "Warning: redundancy columns could not "
            "be identified."
        )

        return pd.DataFrame(
            columns=[
                "feature_a",
                "feature_b",
                "absolute_correlation",
            ]
        )

    result = pd.DataFrame()

    result["feature_a"] = df[feature_a]
    result["feature_b"] = df[feature_b]

    if correlation is not None:

        result[
            "absolute_correlation"
        ] = safe_numeric(
            df[correlation]
        ).abs()

    else:

        result[
            "absolute_correlation"
        ] = np.nan

    return result.dropna(
        subset=[
            "feature_a",
            "feature_b",
        ]
    )


# ======================================================================
# EFFECT CATEGORY
# ======================================================================

def classify_effect(
    effect: Any,
) -> str:

    if effect is None:
        return "unknown"

    try:
        value = abs(float(effect))
    except (
        TypeError,
        ValueError,
    ):
        return "unknown"

    if not math.isfinite(value):
        return "unknown"

    if value >= STRONG_EFFECT_THRESHOLD:
        return "strong"

    if value >= MODERATE_EFFECT_THRESHOLD:
        return "moderate"

    if value >= MIN_EFFECT_THRESHOLD:
        return "small"

    return "negligible"


# ======================================================================
# BUILD CANDIDATE TABLE
# ======================================================================

def build_candidate_table(
    candidates: List[str],
    statistics: pd.DataFrame,
    effects: pd.DataFrame,
    quality_df: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("BUILDING CANDIDATE FEATURE TABLE")
    print("=" * 70)

    base = pd.DataFrame(
        {
            "feature": candidates
        }
    )

    # ================================================================
    # STATISTICAL RESULTS
    # ================================================================

    if not statistics.empty:

        base = base.merge(
            statistics,
            on="feature",
            how="left",
        )

    else:

        base["p_value"] = np.nan
        base["significant"] = False

    # ================================================================
    # EFFECT RESULTS
    # ================================================================

    if not effects.empty:

        base = base.merge(
            effects,
            on="feature",
            how="left",
        )

    else:

        base["effect_size"] = np.nan
        base["absolute_effect"] = np.nan
        base["cliffs_delta"] = np.nan

    # ================================================================
    # QUALITY
    # ================================================================

    if not quality_df.empty:

        base = base.merge(
            quality_df,
            on="feature",
            how="left",
        )

    # ================================================================
    # FALLBACK COLUMNS
    # ================================================================

    ensure_column(
        base,
        "p_value",
        np.nan
    )

    ensure_column(
        base,
        "significant",
        False
    )

    ensure_column(
        base,
        "effect_size",
        np.nan
    )

    ensure_column(
        base,
        "absolute_effect",
        np.nan
    )

    ensure_column(
        base,
        "cliffs_delta",
        np.nan
    )

    # ================================================================
    # NORMALIZE NUMBERS
    # ================================================================

    for column in [
        "p_value",
        "effect_size",
        "absolute_effect",
        "cliffs_delta",
    ]:

        base[column] = safe_numeric(
            base[column]
        )

    # ================================================================
    # EFFECT FALLBACK
    # ================================================================

    base["absolute_effect"] = (
        base["absolute_effect"]
        .fillna(
            base["effect_size"].abs()
        )
    )

    # ================================================================
    # EFFECT CATEGORY
    # ================================================================

    base["effect_category"] = (
        base["absolute_effect"]
        .apply(classify_effect)
    )

    # ================================================================
    # STATISTICAL SIGNIFICANCE
    # ================================================================

    base["significant"] = (
        base["significant"]
        .fillna(False)
        .astype(bool)
    )

    # ================================================================
    # QUALITY SCORE
    # ================================================================

    if "quality_score" not in base.columns:

        base["quality_score"] = 1.0

    base["quality_score"] = (
        safe_numeric(
            base["quality_score"]
        )
        .fillna(0.0)
    )

    return base


# ======================================================================
# REDUNDANCY MAP
# ======================================================================

def build_redundancy_map(
    redundancy: pd.DataFrame,
) -> Dict[str, List[Tuple[str, float]]]:

    result: Dict[
        str,
        List[Tuple[str, float]]
    ] = {}

    if redundancy.empty:
        return result

    for _, row in redundancy.iterrows():

        a = str(
            row["feature_a"]
        )

        b = str(
            row["feature_b"]
        )

        correlation = row[
            "absolute_correlation"
        ]

        if pd.isna(correlation):
            continue

        correlation = float(
            correlation
        )

        result.setdefault(
            a,
            []
        ).append(
            (
                b,
                correlation
            )
        )

        result.setdefault(
            b,
            []
        ).append(
            (
                a,
                correlation
            )
        )

    return result


# ======================================================================
# REDUNDANCY CHECK
# ======================================================================

def is_redundant(
    feature: str,
    selected: List[str],
    redundancy_map: Dict[
        str,
        List[Tuple[str, float]]
    ],
) -> Tuple[bool, Optional[str], float]:

    for selected_feature in selected:

        pairs = redundancy_map.get(
            feature,
            []
        )

        for other, correlation in pairs:

            if other == selected_feature:

                return (
                    correlation >= REDUNDANCY_THRESHOLD,
                    selected_feature,
                    correlation,
                )

    return (
        False,
        None,
        0.0,
    )


# ======================================================================
# FEATURE PRIORITY
# ======================================================================

def feature_priority(
    row: pd.Series,
) -> float:

    effect = row.get(
        "absolute_effect",
        np.nan
    )

    quality = row.get(
        "quality_score",
        0.0
    )

    significant = bool(
        row.get(
            "significant",
            False
        )
    )

    if pd.isna(effect):
        effect = 0.0

    if pd.isna(quality):
        quality = 0.0

    score = (
        float(effect) * 10.0
        + float(quality)
    )

    if significant:
        score += 2.0

    return score


# ======================================================================
# SELECT FEATURES
# ======================================================================

def select_features(
    candidate_df: pd.DataFrame,
    redundancy_map: Dict[
        str,
        List[Tuple[str, float]]
    ],
) -> Tuple[
    List[str],
    List[Dict[str, Any]]
]:

    print()
    print("=" * 70)
    print("REDUNDANCY-AWARE RISK FEATURE SELECTION")
    print("=" * 70)

    df = candidate_df.copy()

    df["priority_score"] = (
        df.apply(
            feature_priority,
            axis=1
        )
    )

    # ================================================================
    # PREFERRED ORDER
    # ================================================================

    preferred_rank = {
        feature: index
        for index, feature
        in enumerate(
            PREFERRED_FEATURES
        )
    }

    df["preferred_rank"] = (
        df["feature"]
        .map(preferred_rank)
        .fillna(9999)
    )

    # ================================================================
    # SORT
    # ================================================================

    df = df.sort_values(
        by=[
            "priority_score",
            "absolute_effect",
            "preferred_rank",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    )

    selected: List[str] = []

    decisions: List[
        Dict[str, Any]
    ] = []

    for _, row in df.iterrows():

        feature = str(
            row["feature"]
        )

        if len(selected) >= MAX_SELECTED_FEATURES:

            decisions.append(
                {
                    "feature": feature,
                    "selected": False,
                    "reason": (
                        "Maximum selected "
                        "feature limit reached."
                    ),
                    "redundant_with": None,
                    "correlation": None,
                }
            )

            continue

        redundant, other, correlation = (
            is_redundant(
                feature,
                selected,
                redundancy_map,
            )
        )

        if redundant:

            decisions.append(
                {
                    "feature": feature,
                    "selected": False,
                    "reason": (
                        "Highly redundant with "
                        "already selected feature."
                    ),
                    "redundant_with": other,
                    "correlation": correlation,
                }
            )

            continue

        # ============================================================
        # SELECT
        # ============================================================

        selected.append(
            feature
        )

        decisions.append(
            {
                "feature": feature,
                "selected": True,
                "reason": (
                    "Retained as a non-redundant "
                    "behavioural risk feature."
                ),
                "redundant_with": None,
                "correlation": None,
            }
        )

    return selected, decisions


# ======================================================================
# CREATE FINAL DATASET
# ======================================================================

def create_selected_dataset(
    research_df: pd.DataFrame,
    selected_features: List[str],
) -> pd.DataFrame:

    columns = [
        "session_id"
    ] + selected_features

    available = [
        column
        for column in columns
        if column in research_df.columns
    ]

    result = research_df[
        available
    ].copy()

    return result


# ======================================================================
# SAVE CSV
# ======================================================================

def save_csv(
    df: pd.DataFrame,
    path: Path,
) -> None:

    df.to_csv(
        path,
        index=False,
    )

    print(
        f"Saved:\n{path}"
    )


# ======================================================================
# SAVE JSON
# ======================================================================

def save_json(
    data: Any,
    path: Path,
) -> None:

    safe_data = sanitize_for_json(
        data
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            safe_data,
            f,
            indent=2,
            allow_nan=False,
        )

    print(
        f"Saved:\n{path}"
    )


# ======================================================================
# QUALITY TABLE
# ======================================================================

def build_quality_table(
    research_df: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    total = len(
        research_df
    )

    for feature in RESEARCH_FEATURES:

        if feature not in research_df.columns:
            continue

        series = safe_numeric(
            research_df[feature]
        )

        missing = int(
            series.isna().sum()
        )

        infinite = int(
            np.isinf(
                series.dropna()
            ).sum()
        )

        unique = int(
            series.nunique(
                dropna=True
            )
        )

        zero_count = int(
            (series == 0).sum()
        )

        if total > 0:

            zero_rate = (
                zero_count
                / total
            )

        else:

            zero_rate = 0.0

        records.append(
            {
                "feature": feature,
                "missing_count": missing,
                "infinite_count": infinite,
                "unique_values": unique,
                "zero_count": zero_count,
                "zero_rate": zero_rate,
                "quality_score": (
                    1.0
                    if missing == 0
                    and infinite == 0
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(
        records
    )


# ======================================================================
# FEATURE REDUNDANCY REPORT
# ======================================================================

def create_selected_redundancy_report(
    selected: List[str],
    redundancy: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    if redundancy.empty:

        return pd.DataFrame(
            columns=[
                "feature_a",
                "feature_b",
                "absolute_correlation",
            ]
        )

    for _, row in redundancy.iterrows():

        a = str(
            row["feature_a"]
        )

        b = str(
            row["feature_b"]
        )

        if (
            a in selected
            and b in selected
        ):

            records.append(
                {
                    "feature_a": a,
                    "feature_b": b,
                    "absolute_correlation": (
                        row[
                            "absolute_correlation"
                        ]
                    ),
                }
            )

    return pd.DataFrame(
        records
    )


# ======================================================================
# BUILD CONFIGURATION
# ======================================================================

def build_configuration(
    selected_features: List[str],
    candidate_df: pd.DataFrame,
    decisions: List[Dict[str, Any]],
    validation_report: Dict[str, Any],
) -> Dict[str, Any]:

    feature_records = []

    lookup = (
        candidate_df
        .set_index("feature")
        .to_dict(
            orient="index"
        )
    )

    for feature in selected_features:

        values = lookup.get(
            feature,
            {}
        )

        feature_records.append(
            {
                "feature": feature,

                "p_value": values.get(
                    "p_value"
                ),

                "significant": values.get(
                    "significant",
                    False
                ),

                "effect_size": values.get(
                    "effect_size"
                ),

                "absolute_effect": values.get(
                    "absolute_effect"
                ),

                "cliffs_delta": values.get(
                    "cliffs_delta"
                ),

                "effect_category": values.get(
                    "effect_category",
                    "unknown"
                ),

                "quality_score": values.get(
                    "quality_score"
                ),
            }
        )

    configuration = {

        "project": "adaptive-deception",

        "stage": "risk_feature_selection",

        "purpose": (
            "Research feature selection for "
            "future behavioural risk scoring."
        ),

        "model_training_completed": False,

        "selected_feature_count": len(
            selected_features
        ),

        "selected_features": selected_features,

        "features": feature_records,

        "selection_configuration": {
            "significance_level": (
                SIGNIFICANCE_LEVEL
            ),

            "minimum_effect_threshold": (
                MIN_EFFECT_THRESHOLD
            ),

            "moderate_effect_threshold": (
                MODERATE_EFFECT_THRESHOLD
            ),

            "strong_effect_threshold": (
                STRONG_EFFECT_THRESHOLD
            ),

            "redundancy_threshold": (
                REDUNDANCY_THRESHOLD
            ),

            "very_high_redundancy_threshold": (
                VERY_HIGH_REDUNDANCY_THRESHOLD
            ),

            "maximum_selected_features": (
                MAX_SELECTED_FEATURES
            ),
        },

        "selection_decisions": decisions,

        "cluster_validation": validation_report,

        "interpretation_warning": (
            "Statistical significance and behavioural "
            "differences do not establish attacker "
            "intent or ground-truth attacker identity."
        ),

        "next_stage": (
            "Risk scoring model design and validation."
        ),
    }

    return configuration


# ======================================================================
# PRINT FINAL SUMMARY
# ======================================================================

def print_final_summary(
    selected_features: List[str],
    candidate_df: pd.DataFrame,
) -> None:

    print()
    print("=" * 70)
    print("RISK FEATURE SELECTION COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Selected features: "
        f"{len(selected_features)}"
    )

    print()

    for index, feature in enumerate(
        selected_features,
        start=1
    ):

        row = candidate_df[
            candidate_df["feature"]
            == feature
        ]

        if len(row) == 0:
            continue

        row = row.iloc[0]

        print(
            f"{index:2d}. "
            f"{feature:35s} "
            f"effect="
            f"{row['absolute_effect']}"
        )

    print()
    print(
        "The selected features are candidates for "
        "the future risk-scoring layer."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Do NOT interpret the selected features as "
        "proof of attacker intent."
    )

    print(
        "Do NOT train the final risk model until "
        "the feature selection has been reviewed."
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)
    print("REDUNDANCY-AWARE RISK FEATURE SELECTION")
    print("=" * 70)

    # ================================================================
    # 1. LOAD RESEARCH DATASET
    # ================================================================

    print()
    print("=" * 70)
    print("LOADING RESEARCH DATASET")
    print("=" * 70)

    research_df = load_csv(
        RESEARCH_DATASET
    )

    validate_research_dataset(
        research_df
    )

    # ================================================================
    # 2. LOAD STATISTICAL TESTS
    # ================================================================

    print()
    print("=" * 70)
    print("LOADING STATISTICAL TEST RESULTS")
    print("=" * 70)

    statistics_raw = load_csv(
        STATISTICAL_TESTS,
        required=False,
    )

    statistics = normalize_statistical_tests(
        statistics_raw
    )

    # ================================================================
    # 3. LOAD EFFECT SIZES
    # ================================================================

    print()
    print("=" * 70)
    print("LOADING EFFECT SIZE RESULTS")
    print("=" * 70)

    effects_raw = load_csv(
        EFFECT_SIZES,
        required=False,
    )

    effects = normalize_effect_sizes(
        effects_raw
    )

    # ================================================================
    # 4. LOAD REDUNDANCY
    # ================================================================

    print()
    print("=" * 70)
    print("LOADING FEATURE REDUNDANCY RESULTS")
    print("=" * 70)

    redundancy_raw = load_csv(
        REDUNDANCY_FILE,
        required=False,
    )

    redundancy = normalize_redundancy(
        redundancy_raw
    )

    # ================================================================
    # 5. LOAD VALIDATION REPORT
    # ================================================================

    validation_report = load_json(
        VALIDATION_REPORT
    )

    # ================================================================
    # 6. QUALITY ANALYSIS
    # ================================================================

    print()
    print("=" * 70)
    print("BUILDING FEATURE QUALITY TABLE")
    print("=" * 70)

    quality_df = build_quality_table(
        research_df
    )

    # ================================================================
    # 7. BUILD CANDIDATE TABLE
    # ================================================================

    candidates = [
        feature
        for feature in RESEARCH_FEATURES
        if feature in research_df.columns
    ]

    candidate_df = build_candidate_table(
        candidates,
        statistics,
        effects,
        quality_df,
    )

    candidate_path = (
        RISK_DIR
        / "risk_feature_candidates.csv"
    )

    save_csv(
        candidate_df,
        candidate_path,
    )

    # ================================================================
    # 8. BUILD REDUNDANCY MAP
    # ================================================================

    redundancy_map = build_redundancy_map(
        redundancy
    )

    # ================================================================
    # 9. SELECT FEATURES
    # ================================================================

    selected_features, decisions = (
        select_features(
            candidate_df,
            redundancy_map,
        )
    )

    # ================================================================
    # 10. CREATE SELECTED DATASET
    # ================================================================

    selected_df = create_selected_dataset(
        research_df,
        selected_features,
    )

    selected_dataset_path = (
        RISK_DIR
        / "selected_risk_features.csv"
    )

    save_csv(
        selected_df,
        selected_dataset_path,
    )

    # ================================================================
    # 11. REDUNDANCY REPORT
    # ================================================================

    selected_redundancy = (
        create_selected_redundancy_report(
            selected_features,
            redundancy,
        )
    )

    redundancy_output = (
        RISK_DIR
        / "risk_feature_redundancy.csv"
    )

    save_csv(
        selected_redundancy,
        redundancy_output,
    )

    # ================================================================
    # 12. CONFIGURATION
    # ================================================================

    configuration = build_configuration(
        selected_features,
        candidate_df,
        decisions,
        validation_report,
    )

    configuration_path = (
        RISK_DIR
        / "risk_feature_configuration.json"
    )

    save_json(
        configuration,
        configuration_path,
    )

    # ================================================================
    # 13. SELECTION REPORT
    # ================================================================

    report = {

        "dataset": {
            "input": str(
                RESEARCH_DATASET
            ),

            "rows": len(
                research_df
            ),

            "candidate_features": len(
                candidates
            ),

            "selected_features": len(
                selected_features
            ),
        },

        "selected_features": selected_features,

        "selection_method": (
            "Effect-size-aware and "
            "redundancy-aware feature selection."
        ),

        "redundancy_threshold": (
            REDUNDANCY_THRESHOLD
        ),

        "maximum_features": (
            MAX_SELECTED_FEATURES
        ),

        "decisions": decisions,

        "outputs": {
            "candidate_features": str(
                candidate_path
            ),

            "selected_features": str(
                selected_dataset_path
            ),

            "redundancy": str(
                redundancy_output
            ),

            "configuration": str(
                configuration_path
            ),
        },

        "warning": (
            "Selected behavioural features do not "
            "constitute ground-truth attacker labels "
            "and do not independently establish "
            "attacker intent."
        ),

        "next_stage": (
            "Design and validate the risk-scoring layer."
        ),
    }

    report_path = (
        RISK_DIR
        / "risk_feature_selection_report.json"
    )

    save_json(
        report,
        report_path,
    )

    # ================================================================
    # 14. FINAL SUMMARY
    # ================================================================

    print_final_summary(
        selected_features,
        candidate_df,
    )

    print()
    print("=" * 70)
    print("GENERATED FILES")
    print("=" * 70)

    print(
        f"1. {candidate_path}"
    )

    print(
        f"2. {selected_dataset_path}"
    )

    print(
        f"3. {redundancy_output}"
    )

    print(
        f"4. {configuration_path}"
    )

    print(
        f"5. {report_path}"
    )

    print()
    print("=" * 70)
    print("DO NOT TRAIN THE RISK MODEL YET")
    print("=" * 70)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()