from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "public"
    / "cic_ids2017"
    / "public_behaviour_dataset.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "public"
    / "cic_ids2017"
)

OUTPUT_FILE = OUTPUT_DIR / "public_scored_dataset.csv"
CONFIG_FILE = OUTPUT_DIR / "public_risk_configuration.json"
REPORT_FILE = OUTPUT_DIR / "public_risk_report.json"


# ============================================================================
# CONFIGURATION
# ============================================================================

RISK_RANGE_MIN = 0.0
RISK_RANGE_MAX = 100.0

RISK_BANDS = [
    {
        "name": "LOW",
        "minimum": 0.0,
        "maximum": 19.999999,
    },
    {
        "name": "GUARDED",
        "minimum": 20.0,
        "maximum": 39.999999,
    },
    {
        "name": "ELEVATED",
        "minimum": 40.0,
        "maximum": 59.999999,
    },
    {
        "name": "HIGH",
        "minimum": 60.0,
        "maximum": 79.999999,
    },
    {
        "name": "CRITICAL",
        "minimum": 80.0,
        "maximum": 100.0,
    },
]


# ============================================================================
# PUBLIC DATASET FEATURES
# ============================================================================

PUBLIC_FEATURES = [
    "event_count",
    "average_event_interval",
    "event_interval_variance",
    "flow_packets_per_sec",
    "flow_bytes_per_sec",
    "forward_packets_per_sec",
    "backward_packets_per_sec",
    "packet_length_mean",
    "packet_length_std",
    "packet_length_variance",
    "max_packet_length",
    "min_packet_length",
    "syn_flag_count",
    "fin_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "ack_flag_count",
    "urg_flag_count",
    "destination_port",
    "network_flag_activity",
    "packet_activity_intensity",
    "directional_packet_balance",
    "connection_reset_ratio",
    "connection_attempt_ratio",
]


# ============================================================================
# FEATURE WEIGHTS
# ============================================================================
#
# These weights are intentionally transparent.
#
# They are NOT copied from the internal 12-feature behavioural score because
# CIC-IDS2017 does not contain the original SSH/session features.
#
# The weights represent the relative importance assigned to network-behaviour
# indicators for this external validation dataset.
#
# They must NOT be interpreted as learned attacker probabilities.
# ============================================================================

FEATURE_WEIGHTS: Dict[str, float] = {
    "event_count": 0.030,

    "average_event_interval": 0.025,
    "event_interval_variance": 0.070,

    "flow_packets_per_sec": 0.080,
    "flow_bytes_per_sec": 0.080,

    "forward_packets_per_sec": 0.050,
    "backward_packets_per_sec": 0.050,

    "packet_length_mean": 0.030,
    "packet_length_std": 0.040,
    "packet_length_variance": 0.040,

    "max_packet_length": 0.025,
    "min_packet_length": 0.015,

    "syn_flag_count": 0.070,
    "fin_flag_count": 0.025,
    "rst_flag_count": 0.070,
    "psh_flag_count": 0.030,
    "ack_flag_count": 0.020,
    "urg_flag_count": 0.015,

    "destination_port": 0.020,

    "network_flag_activity": 0.080,
    "packet_activity_intensity": 0.080,

    "directional_packet_balance": 0.040,

    "connection_reset_ratio": 0.070,
    "connection_attempt_ratio": 0.070,
}


# ============================================================================
# FEATURE DIRECTIONS
# ============================================================================
#
# +1 = larger values increase behavioural risk contribution
# -1 = larger values decrease behavioural risk contribution
#
# These are analytical directions, NOT attacker labels.
# ============================================================================

FEATURE_DIRECTIONS: Dict[str, int] = {
    "event_count": 1,
    "average_event_interval": -1,
    "event_interval_variance": 1,

    "flow_packets_per_sec": 1,
    "flow_bytes_per_sec": 1,

    "forward_packets_per_sec": 1,
    "backward_packets_per_sec": 1,

    "packet_length_mean": 1,
    "packet_length_std": 1,
    "packet_length_variance": 1,

    "max_packet_length": 1,
    "min_packet_length": 1,

    "syn_flag_count": 1,
    "fin_flag_count": 1,
    "rst_flag_count": 1,
    "psh_flag_count": 1,
    "ack_flag_count": 1,
    "urg_flag_count": 1,

    "destination_port": 1,

    "network_flag_activity": 1,
    "packet_activity_intensity": 1,

    "directional_packet_balance": -1,

    "connection_reset_ratio": 1,
    "connection_attempt_ratio": 1,
}


# ============================================================================
# UTILITIES
# ============================================================================


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def validate_configuration() -> None:
    """Validate feature weights and directions."""

    missing_weights = [
        feature
        for feature in PUBLIC_FEATURES
        if feature not in FEATURE_WEIGHTS
    ]

    if missing_weights:
        raise ValueError(
            "Missing weights for features: "
            + ", ".join(missing_weights)
        )

    missing_directions = [
        feature
        for feature in PUBLIC_FEATURES
        if feature not in FEATURE_DIRECTIONS
    ]

    if missing_directions:
        raise ValueError(
            "Missing directions for features: "
            + ", ".join(missing_directions)
        )

    invalid_directions = [
        feature
        for feature in PUBLIC_FEATURES
        if FEATURE_DIRECTIONS[feature] not in (-1, 1)
    ]

    if invalid_directions:
        raise ValueError(
            "Invalid feature direction for: "
            + ", ".join(invalid_directions)
        )

    total_weight = sum(
        FEATURE_WEIGHTS[feature]
        for feature in PUBLIC_FEATURES
    )

    if total_weight <= 0:
        raise ValueError("Total feature weight must be positive.")


def load_dataset() -> pd.DataFrame:
    banner("LOADING PUBLIC BEHAVIOURAL DATASET")

    print(f"Loading:\n{INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Public behavioural dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


def validate_input(df: pd.DataFrame) -> None:
    banner("VALIDATING PUBLIC RISK FEATURES")

    required = PUBLIC_FEATURES + [
        "label",
        "is_attack",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Required columns are missing:\n"
            + "\n".join(f"- {column}" for column in missing)
        )

    print("All required columns are present.")

    invalid_labels = set(
        pd.Series(df["is_attack"])
        .dropna()
        .unique()
    ) - {0, 1}

    if invalid_labels:
        raise ValueError(
            f"'is_attack' contains unexpected values: {invalid_labels}"
        )

    print("Ground-truth columns validated.")


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    banner("CLEANING PUBLIC RISK FEATURES")

    result = df.copy()

    before_missing = int(
        result[PUBLIC_FEATURES]
        .isna()
        .sum()
        .sum()
    )

    before_infinite = int(
        np.isinf(
            result[PUBLIC_FEATURES]
            .to_numpy(dtype=np.float64)
        )
        .sum()
    )

    print(
        f"Missing values before cleaning: "
        f"{before_missing:,}"
    )

    print(
        f"Infinite values before cleaning: "
        f"{before_infinite:,}"
    )

    for feature in PUBLIC_FEATURES:
        result[feature] = pd.to_numeric(
            result[feature],
            errors="coerce",
        )

        result[feature] = result[feature].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        median = result[feature].median()

        if pd.isna(median):
            median = 0.0

        result[feature] = result[feature].fillna(median)

    after_missing = int(
        result[PUBLIC_FEATURES]
        .isna()
        .sum()
        .sum()
    )

    after_infinite = int(
        np.isinf(
            result[PUBLIC_FEATURES]
            .to_numpy(dtype=np.float64)
        )
        .sum()
    )

    print(
        f"Missing values after cleaning: "
        f"{after_missing:,}"
    )

    print(
        f"Infinite values after cleaning: "
        f"{after_infinite:,}"
    )

    return result


# ============================================================================
# ROBUST NORMALIZATION
# ============================================================================


def robust_percentile_normalize(
    series: pd.Series,
) -> pd.Series:
    """
    Convert a numeric feature to [0, 1] using percentile ranking.

    This reduces sensitivity to extreme CIC-IDS2017 outliers.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    values = values.fillna(values.median())

    ranks = values.rank(
        method="average",
        pct=True,
    )

    ranks = ranks.clip(
        lower=0.0,
        upper=1.0,
    )

    return ranks.astype(np.float64)


def build_normalized_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    banner("BUILDING PUBLIC NORMALIZED FEATURES")

    normalized = pd.DataFrame(
        index=df.index
    )

    for feature in PUBLIC_FEATURES:
        normalized_feature = robust_percentile_normalize(
            df[feature]
        )

        normalized[feature] = normalized_feature

        print(f"Processed: {feature}")

    return normalized


# ============================================================================
# FEATURE COMPONENTS
# ============================================================================


def build_components(
    normalized: pd.DataFrame,
) -> pd.DataFrame:
    banner("BUILDING PUBLIC RISK COMPONENTS")

    components = pd.DataFrame(
        index=normalized.index
    )

    normalized_weights = {
        feature: FEATURE_WEIGHTS[feature]
        for feature in PUBLIC_FEATURES
    }

    total_weight = sum(normalized_weights.values())

    for feature in PUBLIC_FEATURES:
        weight = (
            normalized_weights[feature]
            / total_weight
        )

        direction = FEATURE_DIRECTIONS[feature]

        contribution = (
            normalized[feature]
            * weight
            * direction
        )

        # Convert [-weight, +weight] into [0, weight].
        #
        # This keeps every component non-negative and makes the
        # final score easier to interpret.
        component = (
            contribution + weight
        ) / 2.0

        components[
            f"public_component_{feature}"
        ] = component

    return components


# ============================================================================
# RISK SCORE
# ============================================================================


def calculate_risk_score(
    normalized: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    banner("CALCULATING PUBLIC BEHAVIOURAL RISK SCORE")

    total_weight = sum(
        FEATURE_WEIGHTS[feature]
        for feature in PUBLIC_FEATURES
    )

    weighted_values = pd.DataFrame(
        index=normalized.index
    )

    for feature in PUBLIC_FEATURES:
        weight = (
            FEATURE_WEIGHTS[feature]
            / total_weight
        )

        direction = FEATURE_DIRECTIONS[feature]

        weighted_values[feature] = (
            normalized[feature]
            * weight
            * direction
        )

    # Raw score is in [-1, +1].
    raw_score = weighted_values.sum(axis=1)

    # Convert [-1,+1] → [0,100].
    score = (
        (raw_score + 1.0)
        / 2.0
        * 100.0
    )

    score = score.clip(
        lower=RISK_RANGE_MIN,
        upper=RISK_RANGE_MAX,
    )

    score = score.astype(np.float64)

    return score, weighted_values


def assign_risk_band(
    scores: pd.Series,
) -> pd.Series:
    conditions = [
        scores < 20,
        scores < 40,
        scores < 60,
        scores < 80,
        scores <= 100,
    ]

    choices = [
        "LOW",
        "GUARDED",
        "ELEVATED",
        "HIGH",
        "CRITICAL",
    ]

    return pd.Series(
        np.select(
            conditions,
            choices,
            default="CRITICAL",
        ),
        index=scores.index,
        dtype="string",
    )


# ============================================================================
# CONTRIBUTIONS
# ============================================================================


def build_contribution_columns(
    normalized: pd.DataFrame,
) -> pd.DataFrame:
    banner("BUILDING PUBLIC FEATURE CONTRIBUTIONS")

    total_weight = sum(
        FEATURE_WEIGHTS[feature]
        for feature in PUBLIC_FEATURES
    )

    contributions = pd.DataFrame(
        index=normalized.index
    )

    for feature in PUBLIC_FEATURES:
        weight = (
            FEATURE_WEIGHTS[feature]
            / total_weight
        )

        direction = FEATURE_DIRECTIONS[feature]

        contribution = (
            normalized[feature]
            * weight
            * direction
        )

        # Contribution in score points.
        score_contribution = (
            contribution * 50.0
        )

        contributions[
            f"public_contribution_{feature}"
        ] = score_contribution

        print(f"Processed: {feature}")

    return contributions


# ============================================================================
# VALIDATION
# ============================================================================


def validate_output(
    df: pd.DataFrame,
) -> None:
    banner("FINAL PUBLIC RISK SCORE VALIDATION")

    scores = df["public_risk_score"]

    if scores.isna().any():
        raise ValueError(
            "public_risk_score contains missing values."
        )

    if np.isinf(scores.to_numpy()).any():
        raise ValueError(
            "public_risk_score contains infinite values."
        )

    if not scores.between(
        RISK_RANGE_MIN,
        RISK_RANGE_MAX,
    ).all():
        raise ValueError(
            "public_risk_score contains values outside 0-100."
        )

    print(f"Rows: {len(df):,}")
    print(
        f"Minimum score: "
        f"{scores.min():.4f}"
    )
    print(
        f"Maximum score: "
        f"{scores.max():.4f}"
    )
    print(
        f"Mean score: "
        f"{scores.mean():.4f}"
    )
    print(
        f"Median score: "
        f"{scores.median():.4f}"
    )

    print()
    print("Risk-band distribution:")

    distribution = (
        df["public_risk_band"]
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

    for band, count in distribution.items():
        percentage = (
            count
            / len(df)
            * 100.0
        )

        print(
            f"  {band:<10}"
            f"{count:>10,}"
            f" ({percentage:.2f}%)"
        )


# ============================================================================
# REPORT
# ============================================================================


def build_report(
    df: pd.DataFrame,
) -> dict:

    scores = df["public_risk_score"]

    distribution = (
        df["public_risk_band"]
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

    band_distribution = []

    for band, count in distribution.items():
        band_distribution.append(
            {
                "risk_band": band,
                "sessions": int(count),
                "percentage": float(
                    count / len(df) * 100.0
                ),
            }
        )

    attack_distribution = (
        df.groupby("is_attack")
        .size()
        .to_dict()
    )

    label_distribution = (
        df["label"]
        .value_counts()
        .to_dict()
    )

    return {
        "dataset": {
            "name": "CIC-IDS2017",
            "rows": int(len(df)),
            "features": PUBLIC_FEATURES,
            "ground_truth_columns": [
                "label",
                "is_attack",
            ],
        },

        "score": {
            "minimum": float(scores.min()),
            "maximum": float(scores.max()),
            "mean": float(scores.mean()),
            "median": float(scores.median()),
            "range": [
                RISK_RANGE_MIN,
                RISK_RANGE_MAX,
            ],
        },

        "risk_band_distribution":
            band_distribution,

        "ground_truth_distribution": {
            "binary": {
                str(k): int(v)
                for k, v in attack_distribution.items()
            },
            "labels": {
                str(k): int(v)
                for k, v in label_distribution.items()
            },
        },

        "feature_weights": {
            feature: float(
                FEATURE_WEIGHTS[feature]
            )
            for feature in PUBLIC_FEATURES
        },

        "feature_directions": {
            feature: int(
                FEATURE_DIRECTIONS[feature]
            )
            for feature in PUBLIC_FEATURES
        },

        "methodology": {
            "name":
                "Transparent Public Network Behavioural Risk Score",

            "type":
                "unsupervised_external_validation_score",

            "normalization":
                "rank_percentile",

            "score_range":
                [
                    RISK_RANGE_MIN,
                    RISK_RANGE_MAX,
                ],

            "ground_truth_used_for_scoring":
                False,

            "attacker_classifier":
                False,

            "attacker_intent_established":
                False,

            "unavailable_original_features_fabricated":
                False,
        },

        "research_constraints": [
            "CIC-IDS2017 is primarily a network-flow dataset.",
            "SSH usernames are unavailable.",
            "SSH authentication success/failure is unavailable.",
            "Shell commands are unavailable.",
            "File operations are unavailable.",
            "HASSH is unavailable.",
            "Unavailable behavioural features were not fabricated.",
            "CIC-IDS2017 labels are retained as independent ground truth.",
            "The score is not a supervised attacker classifier.",
            "The score does not establish attacker intent.",
            "Adaptive deception deployment requires additional validation.",
        ],
    }


def build_configuration() -> dict:
    total_weight = sum(
        FEATURE_WEIGHTS[feature]
        for feature in PUBLIC_FEATURES
    )

    return {
        "version": "1.0",

        "method": {
            "name":
                "Transparent Public Network Behavioural Risk Score",

            "type":
                "unsupervised_external_validation_score",

            "score_range":
                [
                    RISK_RANGE_MIN,
                    RISK_RANGE_MAX,
                ],

            "normalization":
                "rank_percentile",

            "weight_source":
                "transparent_research_configuration",

            "ground_truth_used_for_scoring":
                False,
        },

        "selected_features":
            PUBLIC_FEATURES,

        "feature_weights": {
            feature: float(
                FEATURE_WEIGHTS[feature]
                / total_weight
            )
            for feature in PUBLIC_FEATURES
        },

        "feature_directions": {
            feature: int(
                FEATURE_DIRECTIONS[feature]
            )
            for feature in PUBLIC_FEATURES
        },

        "risk_bands":
            RISK_BANDS,

        "interpretation": {
            "meaning":
                "Network behavioural separation score derived "
                "from CIC-IDS2017 flow features.",

            "not_ground_truth":
                True,

            "not_attacker_classifier":
                True,

            "attacker_intent_not_established":
                True,

            "deployment_ready":
                False,
        },
    }


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:

    banner("PUBLIC CIC-IDS2017 RISK SCORING")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_configuration()

    print(
        f"Input:\n{INPUT_FILE}"
    )

    print(
        f"Output directory:\n{OUTPUT_DIR}"
    )

    df = load_dataset()

    validate_input(df)

    df = clean_features(df)

    normalized = build_normalized_features(df)

    components = build_components(normalized)

    risk_score, weighted_values = (
        calculate_risk_score(normalized)
    )

    risk_band = assign_risk_band(
        risk_score
    )

    contributions = (
        build_contribution_columns(
            normalized
        )
    )

    # ========================================================================
    # BUILD OUTPUT
    # ========================================================================

    banner("BUILDING PUBLIC SCORED DATASET")

    output = pd.DataFrame(
        index=df.index
    )

    # Preserve identifiers / ground truth.
    if "public_row_id" in df.columns:
        output["public_row_id"] = (
            df["public_row_id"]
        )
    else:
        output["public_row_id"] = (
            np.arange(len(df))
        )

    output["label"] = df["label"]
    output["is_attack"] = df["is_attack"]

    # Preserve original public behavioural features.
    for feature in PUBLIC_FEATURES:
        output[feature] = df[feature]

    # Normalized values.
    for feature in PUBLIC_FEATURES:
        output[
            f"public_normalized_{feature}"
        ] = normalized[feature]

    # Components expected by validator.
    for column in components.columns:
        output[column] = components[column]

    # Feature contribution columns.
    for column in contributions.columns:
        output[column] = contributions[column]

    output["public_risk_score"] = risk_score
    output["public_risk_band"] = risk_band

    # Put important columns near the beginning.
    preferred = [
        "public_row_id",
        "label",
        "is_attack",
        "public_risk_score",
        "public_risk_band",
    ]

    remaining = [
        column
        for column in output.columns
        if column not in preferred
    ]

    output = output[
        preferred + remaining
    ]

    validate_output(output)

    # ========================================================================
    # SAVE DATASET
    # ========================================================================

    banner("SAVING PUBLIC SCORED DATASET")

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Saved:\n{OUTPUT_FILE}"
    )

    # ========================================================================
    # SAVE CONFIGURATION
    # ========================================================================

    banner("SAVING PUBLIC RISK CONFIGURATION")

    configuration = build_configuration()

    with open(
        CONFIG_FILE,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            configuration,
            handle,
            indent=2,
            allow_nan=False,
        )

    print(
        f"Saved:\n{CONFIG_FILE}"
    )

    # ========================================================================
    # SAVE REPORT
    # ========================================================================

    banner("SAVING PUBLIC RISK REPORT")

    report = build_report(output)

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
            allow_nan=False,
        )

    print(
        f"Saved:\n{REPORT_FILE}"
    )

    # ========================================================================
    # COMPLETE
    # ========================================================================

    banner("PUBLIC RISK SCORING COMPLETE")

    print(
        f"Rows scored: {len(output):,}"
    )

    print(
        "Risk score range: 0-100"
    )

    print()
    print(
        "Ground-truth labels were NOT used "
        "to calculate the risk score."
    )

    print()
    print("Generated files:")

    print(
        f"1. {OUTPUT_FILE}"
    )

    print(
        f"2. {CONFIG_FILE}"
    )

    print(
        f"3. {REPORT_FILE}"
    )

    print()
    print("=" * 70)
    print("IMPORTANT RESEARCH CONSTRAINTS")
    print("=" * 70)

    print(
        "1. This is an unsupervised network-behaviour score."
    )

    print(
        "2. CIC-IDS2017 labels were not used to construct the score."
    )

    print(
        "3. The score is NOT an attacker classifier."
    )

    print(
        "4. The score does NOT establish attacker intent."
    )

    print(
        "5. Unavailable SSH/command/file/HASSH features were not fabricated."
    )

    print(
        "6. Ground-truth validation is performed in the next stage."
    )

    print(
        "7. Do NOT deploy adaptive deception policies yet."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()