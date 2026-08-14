"""
define_risk_score.py

Build a transparent behavioural risk-scoring layer from the reviewed
risk features.

IMPORTANT:
- This is NOT a supervised attacker classifier.
- The dataset contains no verified attacker ground-truth labels.
- The resulting score represents behavioural risk/separation only.
- Cluster membership is contextual information, not attacker truth.

Input:
    data/processed/risk/risk_dataset.csv
    data/processed/risk/risk_feature_review.csv

Outputs:
    data/processed/risk/risk_scored_dataset.csv
    data/processed/risk/risk_score_configuration.json
    data/processed/risk/risk_score_feature_contributions.csv
    data/processed/risk/risk_score_report.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ======================================================================
# PATHS
# ======================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RISK_DIR = BASE_DIR / "data" / "processed" / "risk"

RISK_DATASET_FILE = RISK_DIR / "risk_dataset.csv"
FEATURE_REVIEW_FILE = RISK_DIR / "risk_feature_review.csv"

OUTPUT_DATASET_FILE = RISK_DIR / "risk_scored_dataset.csv"
CONFIGURATION_FILE = RISK_DIR / "risk_score_configuration.json"
CONTRIBUTIONS_FILE = RISK_DIR / "risk_score_feature_contributions.csv"
REPORT_FILE = RISK_DIR / "risk_score_report.json"


# ======================================================================
# CONFIGURATION
# ======================================================================

# Features selected during the previous research stage.
SELECTED_FEATURES = [
    "event_interval_variance",
    "duration_sec",
    "failed_auth_ratio",
    "time_to_failed_login_sec",
    "successful_login_count",
    "session_stage_encoded",
    "unique_event_types",
    "unique_event_transitions",
    "unique_hassh",
    "unique_usernames",
    "num_file_events",
    "time_to_first_command_sec",
]


# These directions are based on the reviewed cluster effect signs.
#
# +1:
# Higher values contribute toward behavioural risk.
#
# -1:
# Higher values contribute away from the risk profile represented
# by the selected behavioural separation.
#
# IMPORTANT:
# These are NOT attacker/benign labels.
FEATURE_DIRECTIONS = {
    "event_interval_variance": 1,
    "duration_sec": 1,
    "failed_auth_ratio": 1,
    "time_to_failed_login_sec": 1,
    "successful_login_count": -1,
    "session_stage_encoded": -1,
    "unique_event_types": -1,
    "unique_event_transitions": -1,
    "unique_hassh": -1,
    "unique_usernames": 1,
    "num_file_events": 1,
    "time_to_first_command_sec": -1,
}


# Initial research weights.
#
# These are derived from the absolute reviewed effect sizes, then
# normalized so that the total weight equals 1.
#
# The values are deliberately kept in configuration rather than
# hard-coded into the scoring formula.
DEFAULT_EFFECT_SIZES = {
    "event_interval_variance": 1.000000,
    "duration_sec": 0.968134,
    "failed_auth_ratio": 0.670246,
    "time_to_failed_login_sec": 0.655354,
    "successful_login_count": 0.646532,
    "session_stage_encoded": 0.525433,
    "unique_event_types": 0.376937,
    "unique_event_transitions": 0.325148,
    "unique_hassh": 0.031453,
    "unique_usernames": 0.023911,
    "num_file_events": 0.005111,
    "time_to_first_command_sec": 0.000607,
}


# Features with extremely sparse observations should not dominate
# the score simply because of numerical scaling.
SPARSE_ZERO_RATE_THRESHOLD = 0.95


# Risk bands.
RISK_BANDS = [
    {
        "name": "LOW",
        "minimum": 0,
        "maximum": 19.999999,
    },
    {
        "name": "GUARDED",
        "minimum": 20,
        "maximum": 39.999999,
    },
    {
        "name": "ELEVATED",
        "minimum": 40,
        "maximum": 59.999999,
    },
    {
        "name": "HIGH",
        "minimum": 60,
        "maximum": 79.999999,
    },
    {
        "name": "CRITICAL",
        "minimum": 80,
        "maximum": 100,
    },
]


# ======================================================================
# UTILITIES
# ======================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def ensure_output_directory() -> None:
    RISK_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(value) -> float | None:
    """
    Convert numpy/pandas numeric values into JSON-safe floats.
    """
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return value


def normalize_weights(effect_sizes: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize effect sizes so their sum is 1.
    """

    positive_effects = {
        feature: max(float(effect_sizes.get(feature, 0.0)), 0.0)
        for feature in SELECTED_FEATURES
    }

    total = sum(positive_effects.values())

    if total <= 0:
        raise ValueError(
            "Unable to normalize feature weights because the total "
            "effect size is zero."
        )

    return {
        feature: positive_effects[feature] / total
        for feature in SELECTED_FEATURES
    }


def get_risk_band(score: float) -> str:
    """
    Convert a 0-100 risk score into a qualitative band.
    """

    if score < 20:
        return "LOW"

    if score < 40:
        return "GUARDED"

    if score < 60:
        return "ELEVATED"

    if score < 80:
        return "HIGH"

    return "CRITICAL"


# ======================================================================
# DATA LOADING
# ======================================================================

def load_risk_dataset() -> pd.DataFrame:

    print_header("LOADING RISK DATASET")

    print(f"Loading:")
    print(RISK_DATASET_FILE)

    if not RISK_DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Risk dataset not found:\n{RISK_DATASET_FILE}"
        )

    df = pd.read_csv(RISK_DATASET_FILE)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


def load_feature_review() -> pd.DataFrame:

    print_header("LOADING FEATURE REVIEW")

    print("Loading:")
    print(FEATURE_REVIEW_FILE)

    if not FEATURE_REVIEW_FILE.exists():
        raise FileNotFoundError(
            f"Feature review file not found:\n{FEATURE_REVIEW_FILE}"
        )

    review = pd.read_csv(FEATURE_REVIEW_FILE)

    print(f"Rows: {len(review):,}")
    print(f"Columns: {len(review.columns)}")

    return review


# ======================================================================
# FEATURE VALIDATION
# ======================================================================

def validate_features(df: pd.DataFrame) -> None:

    print_header("VALIDATING RISK FEATURES")

    missing = [
        feature
        for feature in SELECTED_FEATURES
        if feature not in df.columns
    ]

    if missing:
        print("Missing features:")

        for feature in missing:
            print(f" - {feature}")

        raise ValueError(
            "Required risk-scoring features are missing from "
            "risk_dataset.csv."
        )

    print("All selected risk features are present.")

    for feature in SELECTED_FEATURES:

        if not pd.api.types.is_numeric_dtype(df[feature]):
            raise TypeError(
                f"Feature '{feature}' must be numeric."
            )

    print("Numeric validation complete.")


# ======================================================================
# EFFECT SIZE EXTRACTION
# ======================================================================

def extract_effect_sizes(
    review_df: pd.DataFrame,
) -> Dict[str, float]:

    print_header("EXTRACTING REVIEWED EFFECT SIZES")

    effect_sizes = {}

    for feature in SELECTED_FEATURES:

        row = review_df[
            review_df["feature"] == feature
        ]

        if row.empty:

            print(
                f"WARNING: {feature} not found in feature review. "
                f"Using default effect size."
            )

            effect_sizes[feature] = DEFAULT_EFFECT_SIZES[feature]

            continue

        value = row.iloc[0].get("absolute_effect")

        if pd.isna(value):
            value = DEFAULT_EFFECT_SIZES[feature]

        effect_sizes[feature] = float(value)

    print()
    print("Reviewed effect sizes:")

    for feature in SELECTED_FEATURES:
        print(
            f"{feature:35s} "
            f"{effect_sizes[feature]:.6f}"
        )

    return effect_sizes


# ======================================================================
# SPARSITY ANALYSIS
# ======================================================================

def analyse_feature_quality(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print_header("ANALYSING FEATURE QUALITY")

    records = []

    for feature in SELECTED_FEATURES:

        series = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        zero_rate = float(
            (series == 0).mean()
        )

        missing_count = int(
            series.isna().sum()
        )

        infinite_count = int(
            np.isinf(series.to_numpy()).sum()
        )

        unique_values = int(
            series.nunique(dropna=True)
        )

        sparse = (
            zero_rate >= SPARSE_ZERO_RATE_THRESHOLD
        )

        records.append(
            {
                "feature": feature,
                "zero_rate": zero_rate,
                "missing_count": missing_count,
                "infinite_count": infinite_count,
                "unique_values": unique_values,
                "sparse_feature": sparse,
            }
        )

    quality_df = pd.DataFrame(records)

    print(
        quality_df.to_string(index=False)
    )

    return quality_df


# ======================================================================
# SCORE WEIGHTS
# ======================================================================

def build_weights(
    effect_sizes: Dict[str, float],
    quality_df: pd.DataFrame,
) -> Dict[str, float]:

    print_header("BUILDING RISK SCORE WEIGHTS")

    weights = normalize_weights(effect_sizes)

    # We do NOT automatically delete sparse features.
    #
    # Instead, sparse features remain in the transparent score and
    # their quality is explicitly recorded.
    #
    # This avoids silently changing the reviewed feature set.

    print()
    print("Normalized feature weights:")

    for feature in SELECTED_FEATURES:

        print(
            f"{feature:35s} "
            f"{weights[feature]:.6f}"
        )

    print()
    print(
        f"Total weight: {sum(weights.values()):.6f}"
    )

    return weights


# ======================================================================
# FEATURE NORMALIZATION
# ======================================================================

def build_score_components(
    df: pd.DataFrame,
) -> pd.DataFrame:

    """
    Convert each feature into a robust percentile-like 0-1 value.

    Rank-based normalization is used here instead of assuming a
    Gaussian distribution.

    For each feature:

        percentile = rank / (N - 1)

    Then apply the reviewed behavioural direction.

    Direction +1:
        high value -> higher risk contribution

    Direction -1:
        high value -> lower risk contribution
    """

    print_header("BUILDING ROBUST SCORE COMPONENTS")

    components = pd.DataFrame(
        index=df.index
    )

    for feature in SELECTED_FEATURES:

        values = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        values = values.fillna(
            values.median()
        )

        ranks = values.rank(
            method="average",
            pct=True,
        )

        direction = FEATURE_DIRECTIONS[feature]

        if direction == 1:
            component = ranks
        else:
            component = 1.0 - ranks

        component = component.clip(
            lower=0.0,
            upper=1.0,
        )

        components[
            f"component_{feature}"
        ] = component

        print(
            f"Processed: {feature}"
        )

    return components


# ======================================================================
# RISK SCORE
# ======================================================================

def calculate_risk_score(
    components: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.DataFrame:

    print_header("CALCULATING BEHAVIOURAL RISK SCORE")

    score = np.zeros(
        len(components),
        dtype=np.float64,
    )

    contribution_data = {}

    for feature in SELECTED_FEATURES:

        component_column = (
            f"component_{feature}"
        )

        component = components[
            component_column
        ].to_numpy()

        weighted_contribution = (
            component * weights[feature]
        )

        score += weighted_contribution

        contribution_data[
            f"contribution_{feature}"
        ] = (
            weighted_contribution * 100.0
        )

    score = np.clip(
        score * 100.0,
        0.0,
        100.0,
    )

    result = pd.DataFrame(
        contribution_data,
        index=components.index,
    )

    result["risk_score"] = score

    result["risk_band"] = [
        get_risk_band(value)
        for value in score
    ]

    return result


# ======================================================================
# CONTRIBUTION REPORT
# ======================================================================

def build_contribution_report(
    score_df: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.DataFrame:

    print_header("BUILDING FEATURE CONTRIBUTION REPORT")

    records = []

    for feature in SELECTED_FEATURES:

        column = f"contribution_{feature}"

        values = score_df[column]

        records.append(
            {
                "feature": feature,
                "weight": weights[feature],
                "mean_contribution": float(
                    values.mean()
                ),
                "median_contribution": float(
                    values.median()
                ),
                "max_contribution": float(
                    values.max()
                ),
                "min_contribution": float(
                    values.min()
                ),
            }
        )

    report = pd.DataFrame(records)

    report = report.sort_values(
        "weight",
        ascending=False,
    ).reset_index(drop=True)

    return report


# ======================================================================
# DISTRIBUTION REPORT
# ======================================================================

def build_distribution_report(
    scored_df: pd.DataFrame,
) -> Dict:

    scores = scored_df["risk_score"]

    band_counts = (
        scored_df["risk_band"]
        .value_counts()
        .reindex(
            [
                "LOW",
                "GUARDED",
                "ELEVATED",
                "HIGH",
                "CRITICAL",
            ],
            fill_value=0,
        )
    )

    band_distribution = {}

    total = len(scored_df)

    for band, count in band_counts.items():

        count = int(count)

        band_distribution[band] = {
            "sessions": count,
            "percentage": (
                float(count / total * 100.0)
                if total > 0
                else 0.0
            ),
        }

    return {
        "minimum": safe_float(scores.min()),
        "maximum": safe_float(scores.max()),
        "mean": safe_float(scores.mean()),
        "median": safe_float(scores.median()),
        "std": safe_float(scores.std()),
        "percentile_25": safe_float(
            scores.quantile(0.25)
        ),
        "percentile_75": safe_float(
            scores.quantile(0.75)
        ),
        "percentile_95": safe_float(
            scores.quantile(0.95)
        ),
        "percentile_99": safe_float(
            scores.quantile(0.99)
        ),
        "risk_bands": band_distribution,
    }


# ======================================================================
# CONFIGURATION
# ======================================================================

def build_configuration(
    effect_sizes: Dict[str, float],
    weights: Dict[str, float],
    quality_df: pd.DataFrame,
) -> Dict:

    sparse_features = quality_df[
        quality_df["sparse_feature"]
    ]["feature"].tolist()

    return {
        "version": "1.0",
        "method": {
            "name": "Transparent Behavioural Risk Score",
            "type": "unsupervised_behavioural_scoring",
            "score_range": [0, 100],
            "normalization": "rank_percentile",
            "weight_source": "reviewed_absolute_effect_size",
        },
        "selected_features": SELECTED_FEATURES,
        "feature_directions": FEATURE_DIRECTIONS,
        "effect_sizes": effect_sizes,
        "normalized_weights": weights,
        "sparse_features": sparse_features,
        "sparse_zero_rate_threshold": (
            SPARSE_ZERO_RATE_THRESHOLD
        ),
        "risk_bands": RISK_BANDS,
        "interpretation": {
            "meaning": (
                "Behavioural risk/separation score derived "
                "from reviewed session features."
            ),
            "not_ground_truth": True,
            "not_attacker_classifier": True,
            "attacker_intent_not_established": True,
        },
    }


# ======================================================================
# JSON REPORT
# ======================================================================

def save_json(
    path: Path,
    data: Dict,
) -> None:

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            allow_nan=False,
        )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print_header(
        "BEHAVIOURAL RISK SCORE DEFINITION"
    )

    ensure_output_directory()

    # --------------------------------------------------------------
    # Load
    # --------------------------------------------------------------

    df = load_risk_dataset()

    review_df = load_feature_review()

    # --------------------------------------------------------------
    # Validate
    # --------------------------------------------------------------

    validate_features(df)

    # --------------------------------------------------------------
    # Effect sizes
    # --------------------------------------------------------------

    effect_sizes = extract_effect_sizes(
        review_df
    )

    # --------------------------------------------------------------
    # Feature quality
    # --------------------------------------------------------------

    quality_df = analyse_feature_quality(
        df
    )

    # --------------------------------------------------------------
    # Weights
    # --------------------------------------------------------------

    weights = build_weights(
        effect_sizes,
        quality_df,
    )

    # --------------------------------------------------------------
    # Score components
    # --------------------------------------------------------------

    components = build_score_components(
        df
    )

    # --------------------------------------------------------------
    # Calculate score
    # --------------------------------------------------------------

    score_df = calculate_risk_score(
        components,
        weights,
    )

    # --------------------------------------------------------------
    # Build final dataset
    # --------------------------------------------------------------

    scored_df = pd.DataFrame(
        {
            "session_id": df["session_id"],
            "risk_score": score_df["risk_score"],
            "risk_band": score_df["risk_band"],
        }
    )

    # Include individual behavioural components.
    for feature in SELECTED_FEATURES:

        scored_df[
            f"risk_component_{feature}"
        ] = components[
            f"component_{feature}"
        ].to_numpy()

    # Include individual contributions.
    for feature in SELECTED_FEATURES:

        scored_df[
            f"risk_contribution_{feature}"
        ] = score_df[
            f"contribution_{feature}"
        ].to_numpy()

    # --------------------------------------------------------------
    # Validate scores
    # --------------------------------------------------------------

    print_header(
        "FINAL RISK SCORE VALIDATION"
    )

    if scored_df["risk_score"].isna().any():
        raise ValueError(
            "Risk score contains missing values."
        )

    if np.isinf(
        scored_df["risk_score"].to_numpy()
    ).any():

        raise ValueError(
            "Risk score contains infinite values."
        )

    if (
        (scored_df["risk_score"] < 0)
        | (scored_df["risk_score"] > 100)
    ).any():

        raise ValueError(
            "Risk scores must remain within 0-100."
        )

    if scored_df["session_id"].duplicated().any():

        raise ValueError(
            "Duplicate session IDs detected."
        )

    print(
        f"Rows: {len(scored_df):,}"
    )

    print(
        f"Minimum score: "
        f"{scored_df['risk_score'].min():.4f}"
    )

    print(
        f"Maximum score: "
        f"{scored_df['risk_score'].max():.4f}"
    )

    print(
        f"Mean score: "
        f"{scored_df['risk_score'].mean():.4f}"
    )

    print(
        f"Median score: "
        f"{scored_df['risk_score'].median():.4f}"
    )

    # --------------------------------------------------------------
    # Contribution report
    # --------------------------------------------------------------

    contribution_report = (
        build_contribution_report(
            score_df,
            weights,
        )
    )

    # --------------------------------------------------------------
    # Distribution
    # --------------------------------------------------------------

    distribution = build_distribution_report(
        scored_df
    )

    # --------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------

    configuration = build_configuration(
        effect_sizes,
        weights,
        quality_df,
    )

    # --------------------------------------------------------------
    # Research report
    # --------------------------------------------------------------

    report = {
        "status": "research_candidate",
        "rows": int(len(scored_df)),
        "features": SELECTED_FEATURES,
        "score": {
            "minimum": distribution["minimum"],
            "maximum": distribution["maximum"],
            "mean": distribution["mean"],
            "median": distribution["median"],
            "std": distribution["std"],
        },
        "risk_bands": distribution["risk_bands"],
        "feature_weights": weights,
        "feature_quality": quality_df.to_dict(
            orient="records"
        ),
        "limitations": [
            "No verified attacker ground-truth labels "
            "are available.",
            "The score represents behavioural separation "
            "rather than confirmed malicious intent.",
            "Cluster membership is not treated as an "
            "attacker label.",
            "The weighting scheme requires validation "
            "before operational deployment.",
            "Risk thresholds are research thresholds and "
            "must be calibrated against labelled evaluation "
            "data before production use.",
        ],
        "next_stage": [
            "Validate score stability.",
            "Inspect score distribution.",
            "Perform sensitivity analysis.",
            "Evaluate false-positive behaviour.",
            "Calibrate thresholds using labelled data "
            "if ground truth becomes available.",
        ],
    }

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    print_header(
        "SAVING RISK SCORE DATASET"
    )

    scored_df.to_csv(
        OUTPUT_DATASET_FILE,
        index=False,
    )

    print("Saved:")
    print(OUTPUT_DATASET_FILE)

    print_header(
        "SAVING FEATURE CONTRIBUTIONS"
    )

    contribution_report.to_csv(
        CONTRIBUTIONS_FILE,
        index=False,
    )

    print("Saved:")
    print(CONTRIBUTIONS_FILE)

    print_header(
        "SAVING CONFIGURATION"
    )

    save_json(
        CONFIGURATION_FILE,
        configuration,
    )

    print("Saved:")
    print(CONFIGURATION_FILE)

    print_header(
        "SAVING RISK SCORE REPORT"
    )

    save_json(
        REPORT_FILE,
        report,
    )

    print("Saved:")
    print(REPORT_FILE)

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    print_header(
        "RISK SCORE DEFINITION COMPLETE"
    )

    print(
        f"Rows scored: {len(scored_df):,}"
    )

    print(
        "Risk score range: 0-100"
    )

    print()

    print(
        "Risk-band distribution:"
    )

    for band, values in distribution[
        "risk_bands"
    ].items():

        print(
            f"  {band:10s} "
            f"{values['sessions']:>10,} "
            f"({values['percentage']:.2f}%)"
        )

    print()

    print(
        "Top weighted features:"
    )

    for feature, weight in sorted(
        weights.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:8]:

        print(
            f"  {feature:35s} "
            f"{weight:.6f}"
        )

    print()
    print("=" * 70)
    print("IMPORTANT")
    print("=" * 70)
    print(
        "This is a behavioural research score."
    )
    print(
        "It is NOT proof of attacker intent."
    )
    print(
        "It is NOT a supervised attacker classifier."
    )
    print(
        "Do NOT deploy adaptive deception policies yet."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()