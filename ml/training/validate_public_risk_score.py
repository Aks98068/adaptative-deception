from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import mannwhitneyu
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PUBLIC_DATASET = (
    BASE_DIR
    / "data"
    / "processed"
    / "public"
    / "cic_ids2017"
    / "public_behaviour_dataset.csv"
)

RISK_CONFIGURATION = (
    BASE_DIR
    / "data"
    / "processed"
    / "risk"
    / "risk_score_configuration.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "public"
    / "cic_ids2017"
    / "validation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONFIGURATION
# ============================================================================

ID_COLUMN = "public_row_id"
LABEL_COLUMN = "label"
GROUND_TRUTH_COLUMN = "is_attack"

# Features that have a defensible network-flow behavioural interpretation.
#
# IMPORTANT:
# These are NOT being claimed to be equivalent to the original SSH/session
# features. They are a separate public-dataset validation feature space.
#
# The signs below represent the direction in which the feature contributes
# to the public network behavioural score.
#
# This is deliberately transparent and documented rather than pretending that
# CIC-IDS2017 contains the original authentication/command/file features.
PUBLIC_FEATURE_DIRECTIONS = {
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
    "network_flag_activity": 1,
    "packet_activity_intensity": 1,
    "directional_packet_balance": 1,
    "connection_reset_ratio": 1,
    "connection_attempt_ratio": 1,
}

# The destination port is retained for analysis but is NOT included in the
# default behavioural score because port number is categorical/contextual
# rather than intrinsically "more risky" when numerically larger.
#
# destination_port is therefore analysed separately.

PUBLIC_FEATURES = list(PUBLIC_FEATURE_DIRECTIONS.keys())


# Original model features that cannot honestly be obtained from CIC-IDS2017.
UNAVAILABLE_ORIGINAL_FEATURES = [
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

PARTIALLY_OR_NOT_DIRECTLY_MAPPED_ORIGINAL_FEATURES = [
    "event_interval_variance",
    "duration_sec",
]


RISK_BANDS = [
    ("LOW", 0.0, 19.999999),
    ("GUARDED", 20.0, 39.999999),
    ("ELEVATED", 40.0, 59.999999),
    ("HIGH", 60.0, 79.999999),
    ("CRITICAL", 80.0, 100.0),
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def save_json(path: Path, data: dict) -> None:
    def sanitize(value):
        if isinstance(value, dict):
            return {str(k): sanitize(v) for k, v in value.items()}

        if isinstance(value, list):
            return [sanitize(v) for v in value]

        if isinstance(value, tuple):
            return [sanitize(v) for v in value]

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            if not np.isfinite(value):
                return None
            return float(value)

        if isinstance(value, float):
            if not math.isfinite(value):
                return None
            return value

        if pd.isna(value):
            return None

        return value

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            sanitize(data),
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def safe_float(value):
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass

    return None


def robust_normalize(series: pd.Series) -> pd.Series:
    """
    Robust percentile normalization.

    Converts a feature into [0, 1] using empirical percentile ranks.

    This avoids allowing extreme CIC-IDS2017 flow values to dominate the
    score merely because they have large numerical magnitude.
    """

    values = pd.to_numeric(series, errors="coerce")

    if values.notna().sum() == 0:
        return pd.Series(
            np.zeros(len(values), dtype=float),
            index=values.index,
        )

    median = values.median()

    values = values.replace([np.inf, -np.inf], np.nan)
    values = values.fillna(median)

    # Constant feature.
    if values.nunique(dropna=False) <= 1:
        return pd.Series(
            np.zeros(len(values), dtype=float),
            index=values.index,
        )

    ranks = values.rank(method="average", pct=True)

    return ranks.clip(0.0, 1.0)


def assign_risk_band(score: float) -> str:
    if score < 20:
        return "LOW"

    if score < 40:
        return "GUARDED"

    if score < 60:
        return "ELEVATED"

    if score < 80:
        return "HIGH"

    return "CRITICAL"


def calculate_cliffs_delta(
    benign: np.ndarray,
    attack: np.ndarray,
) -> float | None:
    """
    Calculate Cliff's delta:

        delta = P(X_attack > X_benign)
              - P(X_attack < X_benign)

    Uses a rank-based formulation suitable for large datasets.
    """

    benign = np.asarray(benign, dtype=float)
    attack = np.asarray(attack, dtype=float)

    benign = benign[np.isfinite(benign)]
    attack = attack[np.isfinite(attack)]

    if len(benign) == 0 or len(attack) == 0:
        return None

    combined = np.concatenate([benign, attack])

    ranks = pd.Series(combined).rank(method="average").to_numpy()

    attack_ranks = ranks[len(benign):]

    n_attack = len(attack)
    n_benign = len(benign)

    u_attack = (
        attack_ranks.sum()
        - n_attack * (n_attack + 1) / 2
    )

    delta = (
        2.0 * u_attack / (n_attack * n_benign)
        - 1.0
    )

    return float(delta)


# ============================================================================
# LOAD DATA
# ============================================================================

def load_public_dataset() -> pd.DataFrame:
    print_header("LOADING PUBLIC BEHAVIOURAL DATASET")

    print(f"Loading:")
    print(PUBLIC_DATASET)

    if not PUBLIC_DATASET.exists():
        raise FileNotFoundError(
            f"Public behavioural dataset not found:\n{PUBLIC_DATASET}"
        )

    df = pd.read_csv(PUBLIC_DATASET)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


def load_original_configuration() -> dict:
    print_header("LOADING ORIGINAL RISK CONFIGURATION")

    if not RISK_CONFIGURATION.exists():
        raise FileNotFoundError(
            f"Risk configuration not found:\n{RISK_CONFIGURATION}"
        )

    with open(RISK_CONFIGURATION, "r", encoding="utf-8") as f:
        configuration = json.load(f)

    print(f"Loaded:")
    print(RISK_CONFIGURATION)

    return configuration


# ============================================================================
# VALIDATION
# ============================================================================

def validate_schema(df: pd.DataFrame) -> None:
    print_header("VALIDATING PUBLIC DATASET SCHEMA")

    required = {
        ID_COLUMN,
        LABEL_COLUMN,
        GROUND_TRUTH_COLUMN,
        *PUBLIC_FEATURES,
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            "Public behavioural dataset is missing required columns:\n"
            + "\n".join(f"- {x}" for x in missing)
        )

    print("All required columns are present.")

    print()
    print("Ground-truth column:")
    print(f"- {GROUND_TRUTH_COLUMN}")

    print()
    print("Behavioural validation features:")
    for feature in PUBLIC_FEATURES:
        print(f"- {feature}")


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    print_header("CLEANING PUBLIC VALIDATION FEATURES")

    df = df.copy()

    before_missing = int(
        df[PUBLIC_FEATURES]
        .isna()
        .sum()
        .sum()
    )

    before_infinite = int(
        np.isinf(
            df[PUBLIC_FEATURES]
            .to_numpy(dtype=float)
        )
        .sum()
    )

    for feature in PUBLIC_FEATURES:
        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        df[feature] = (
            df[feature]
            .replace([np.inf, -np.inf], np.nan)
        )

        median = df[feature].median()

        if pd.isna(median):
            median = 0.0

        df[feature] = df[feature].fillna(median)

    after_missing = int(
        df[PUBLIC_FEATURES]
        .isna()
        .sum()
        .sum()
    )

    after_infinite = int(
        np.isinf(
            df[PUBLIC_FEATURES]
            .to_numpy(dtype=float)
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

    print(
        f"Missing values after cleaning: "
        f"{after_missing:,}"
    )

    print(
        f"Infinite values after cleaning: "
        f"{after_infinite:,}"
    )

    return df


# ============================================================================
# PUBLIC SCORE
# ============================================================================

def build_public_feature_scores(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    print_header("BUILDING PUBLIC NETWORK BEHAVIOURAL FEATURE SCORES")

    feature_scores = pd.DataFrame(
        index=df.index
    )

    feature_weights = {}

    # Equal initial weighting is deliberate.
    #
    # We are validating the public behavioural representation rather than
    # silently transferring the original SSH risk weights to unrelated
    # network-flow variables.
    equal_weight = 1.0 / len(PUBLIC_FEATURES)

    for feature in PUBLIC_FEATURES:

        normalized = robust_normalize(df[feature])

        direction = PUBLIC_FEATURE_DIRECTIONS[feature]

        if direction < 0:
            normalized = 1.0 - normalized

        column_name = (
            f"public_component_{feature}"
        )

        feature_scores[column_name] = normalized

        feature_weights[feature] = equal_weight

        print(
            f"Processed: {feature}"
            f" | direction={direction:+d}"
        )

    return feature_scores, feature_weights


def calculate_public_risk_score(
    feature_scores: pd.DataFrame,
    feature_weights: Dict[str, float],
) -> pd.Series:
    print_header("CALCULATING PUBLIC BEHAVIOURAL RISK SCORE")

    score = np.zeros(len(feature_scores), dtype=float)

    for feature, weight in feature_weights.items():
        column = f"public_component_{feature}"

        score += (
            feature_scores[column].to_numpy()
            * weight
        )

    # Convert [0,1] to [0,100].
    score *= 100.0

    return pd.Series(
        np.clip(score, 0.0, 100.0),
        index=feature_scores.index,
        name="public_risk_score",
    )


# ============================================================================
# SCORE VALIDATION
# ============================================================================

def validate_scores(
    df: pd.DataFrame,
    score: pd.Series,
) -> dict:

    print_header("VALIDATING PUBLIC RISK SCORES")

    values = score.to_numpy(dtype=float)

    report = {
        "rows": int(len(score)),
        "minimum": safe_float(np.min(values)),
        "maximum": safe_float(np.max(values)),
        "mean": safe_float(np.mean(values)),
        "median": safe_float(np.median(values)),
        "missing": int(score.isna().sum()),
        "infinite": int(
            np.isinf(values).sum()
        ),
    }

    print(
        f"Rows: {report['rows']:,}"
    )

    print(
        f"Minimum score: "
        f"{report['minimum']:.4f}"
    )

    print(
        f"Maximum score: "
        f"{report['maximum']:.4f}"
    )

    print(
        f"Mean score: "
        f"{report['mean']:.4f}"
    )

    print(
        f"Median score: "
        f"{report['median']:.4f}"
    )

    return report


def build_risk_bands(
    score: pd.Series,
) -> pd.Series:

    return score.apply(assign_risk_band)


def calculate_band_distribution(
    bands: pd.Series,
) -> pd.DataFrame:

    counts = (
        bands
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

    total = len(bands)

    result = pd.DataFrame(
        {
            "risk_band": counts.index,
            "sessions": counts.values,
            "percentage": (
                counts.values / total * 100.0
            ),
        }
    )

    return result


# ============================================================================
# GROUND TRUTH ANALYSIS
# ============================================================================

def calculate_ground_truth_distribution(
    df: pd.DataFrame,
) -> pd.DataFrame:

    distribution = (
        df.groupby(
            [LABEL_COLUMN, GROUND_TRUTH_COLUMN],
            dropna=False,
        )
        .size()
        .reset_index(name="sessions")
    )

    total = len(df)

    distribution["percentage"] = (
        distribution["sessions"]
        / total
        * 100.0
    )

    return distribution


def calculate_score_by_ground_truth(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = (
        df.groupby(
            GROUND_TRUTH_COLUMN
        )["public_risk_score"]
        .agg(
            sessions="count",
            mean_risk_score="mean",
            median_risk_score="median",
            std_risk_score="std",
            minimum_risk_score="min",
            maximum_risk_score="max",
        )
        .reset_index()
    )

    return result


def calculate_score_by_label(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = (
        df.groupby(
            LABEL_COLUMN
        )["public_risk_score"]
        .agg(
            sessions="count",
            mean_risk_score="mean",
            median_risk_score="median",
            std_risk_score="std",
            minimum_risk_score="min",
            maximum_risk_score="max",
        )
        .reset_index()
        .sort_values(
            "mean_risk_score",
            ascending=False,
        )
    )

    return result


def calculate_attack_rate_by_band(
    df: pd.DataFrame,
) -> pd.DataFrame:

    grouped = (
        df.groupby("risk_band")
        .agg(
            sessions=(
                GROUND_TRUTH_COLUMN,
                "size",
            ),
            attacks=(
                GROUND_TRUTH_COLUMN,
                "sum",
            ),
        )
        .reindex(
            [
                "LOW",
                "GUARDED",
                "ELEVATED",
                "HIGH",
                "CRITICAL",
            ]
        )
        .fillna(0)
        .reset_index()
    )

    grouped["attack_rate_percentage"] = (
        grouped["attacks"]
        / grouped["sessions"].replace(0, np.nan)
        * 100.0
    )

    grouped["attack_rate_percentage"] = (
        grouped["attack_rate_percentage"]
        .fillna(0.0)
    )

    return grouped


# ============================================================================
# STATISTICAL VALIDATION
# ============================================================================

def calculate_statistical_validation(
    df: pd.DataFrame,
) -> dict:

    print_header("STATISTICAL GROUND-TRUTH VALIDATION")

    benign = (
        df.loc[
            df[GROUND_TRUTH_COLUMN] == 0,
            "public_risk_score",
        ]
        .to_numpy(dtype=float)
    )

    attack = (
        df.loc[
            df[GROUND_TRUTH_COLUMN] == 1,
            "public_risk_score",
        ]
        .to_numpy(dtype=float)
    )

    result = {
        "benign_sessions": int(len(benign)),
        "attack_sessions": int(len(attack)),
        "benign_mean": safe_float(np.mean(benign)),
        "attack_mean": safe_float(np.mean(attack)),
        "benign_median": safe_float(np.median(benign)),
        "attack_median": safe_float(np.median(attack)),
        "mean_difference_attack_minus_benign": safe_float(
            np.mean(attack) - np.mean(benign)
        ),
        "cliffs_delta_attack_vs_benign": calculate_cliffs_delta(
            benign,
            attack,
        ),
    }

    if SCIPY_AVAILABLE:

        # Mann-Whitney U can become extremely significant with huge datasets.
        # Therefore the report emphasizes effect size as well.
        statistic, p_value = mannwhitneyu(
            attack,
            benign,
            alternative="two-sided",
        )

        result["mann_whitney_u"] = safe_float(
            statistic
        )

        result["mann_whitney_p_value"] = safe_float(
            p_value
        )

    else:
        result["mann_whitney_u"] = None
        result["mann_whitney_p_value"] = None

    print(
        f"Benign sessions: {len(benign):,}"
    )

    print(
        f"Attack sessions: {len(attack):,}"
    )

    print(
        f"Benign mean score: "
        f"{np.mean(benign):.4f}"
    )

    print(
        f"Attack mean score: "
        f"{np.mean(attack):.4f}"
    )

    print(
        f"Mean difference: "
        f"{np.mean(attack) - np.mean(benign):.4f}"
    )

    print(
        f"Cliff's delta: "
        f"{result['cliffs_delta_attack_vs_benign']}"
    )

    if result["mann_whitney_p_value"] is not None:
        print(
            f"Mann-Whitney p-value: "
            f"{result['mann_whitney_p_value']:.6e}"
        )

    return result


# ============================================================================
# ROC / PR ANALYSIS
# ============================================================================

def calculate_classification_metrics(
    df: pd.DataFrame,
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:

    print_header("CALCULATING GROUND-TRUTH PERFORMANCE METRICS")

    y_true = df[GROUND_TRUTH_COLUMN].astype(int)
    scores = df["public_risk_score"].astype(float)

    if not SKLEARN_AVAILABLE:
        print(
            "WARNING: scikit-learn is not installed."
        )

        return (
            {
                "roc_auc": None,
                "average_precision": None,
            },
            pd.DataFrame(),
            pd.DataFrame(),
        )

    roc_auc = roc_auc_score(
        y_true,
        scores,
    )

    average_precision = average_precision_score(
        y_true,
        scores,
    )

    fpr, tpr, thresholds = roc_curve(
        y_true,
        scores,
    )

    precision, recall, pr_thresholds = (
        precision_recall_curve(
            y_true,
            scores,
        )
    )

    roc_df = pd.DataFrame(
        {
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds,
        }
    )

    # precision_recall_curve returns one more precision/recall point
    # than threshold values.
    pr_df = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
        }
    )

    print(
        f"ROC-AUC: "
        f"{roc_auc:.6f}"
    )

    print(
        f"PR-AUC / Average Precision: "
        f"{average_precision:.6f}"
    )

    metrics = {
        "roc_auc": safe_float(roc_auc),
        "average_precision": safe_float(
            average_precision
        ),
    }

    return metrics, roc_df, pr_df


# ============================================================================
# THRESHOLD ANALYSIS
# ============================================================================

def calculate_threshold_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print_header("CALCULATING THRESHOLD PERFORMANCE")

    if not SKLEARN_AVAILABLE:
        return pd.DataFrame()

    y_true = df[GROUND_TRUTH_COLUMN].astype(int)
    scores = df["public_risk_score"]

    thresholds = [
        20,
        40,
        60,
        80,
    ]

    rows = []

    for threshold in thresholds:

        predicted_attack = (
            scores >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predicted_attack,
            labels=[0, 1],
        ).ravel()

        total = tn + fp + fn + tp

        accuracy = (
            (tp + tn) / total
            if total
            else 0.0
        )

        precision = (
            tp / (tp + fp)
            if tp + fp
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn
            else 0.0
        )

        specificity = (
            tn / (tn + fp)
            if tn + fp
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if precision + recall
            else 0.0
        )

        rows.append(
            {
                "threshold": threshold,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1_score": f1,
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# FEATURE ANALYSIS
# ============================================================================

def calculate_feature_ground_truth_effects(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print_header("ANALYSING PUBLIC FEATURE SEPARATION")

    rows = []

    benign_mask = (
        df[GROUND_TRUTH_COLUMN] == 0
    )

    attack_mask = (
        df[GROUND_TRUTH_COLUMN] == 1
    )

    for feature in PUBLIC_FEATURES:

        benign = (
            df.loc[
                benign_mask,
                feature,
            ]
            .to_numpy(dtype=float)
        )

        attack = (
            df.loc[
                attack_mask,
                feature,
            ]
            .to_numpy(dtype=float)
        )

        benign_median = np.median(
            benign
        )

        attack_median = np.median(
            attack
        )

        delta = calculate_cliffs_delta(
            benign,
            attack,
        )

        rows.append(
            {
                "feature": feature,
                "direction": (
                    PUBLIC_FEATURE_DIRECTIONS[
                        feature
                    ]
                ),
                "benign_mean": np.mean(
                    benign
                ),
                "attack_mean": np.mean(
                    attack
                ),
                "benign_median": benign_median,
                "attack_median": attack_median,
                "mean_difference": (
                    np.mean(attack)
                    - np.mean(benign)
                ),
                "median_difference": (
                    attack_median
                    - benign_median
                ),
                "cliffs_delta": delta,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "cliffs_delta",
            key=lambda x: x.abs(),
            ascending=False,
        )
    )


# ============================================================================
# ORIGINAL MODEL COMPARISON
# ============================================================================

def build_feature_mapping_report() -> pd.DataFrame:

    rows = []

    original_features = [
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

    for feature in original_features:

        if feature in UNAVAILABLE_ORIGINAL_FEATURES:
            status = "UNAVAILABLE"
            reason = (
                "CIC-IDS2017 does not directly represent "
                "this SSH/session behavioural field."
            )

        elif feature in (
            PARTIALLY_OR_NOT_DIRECTLY_MAPPED_ORIGINAL_FEATURES
        ):
            status = "APPROXIMATE_OR_DERIVED"
            reason = (
                "A related network-flow representation exists, "
                "but it is not semantically identical to the "
                "original behavioural feature."
            )

        else:
            status = "REVIEW"
            reason = (
                "Requires semantic review before claiming "
                "feature equivalence."
            )

        rows.append(
            {
                "original_risk_feature": feature,
                "status": status,
                "reason": reason,
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# REPORT
# ============================================================================

def build_validation_report(
    df: pd.DataFrame,
    score_validation: dict,
    statistical_validation: dict,
    classification_metrics: dict,
    band_distribution: pd.DataFrame,
    attack_rate_by_band: pd.DataFrame,
    feature_effects: pd.DataFrame,
    mapping_report: pd.DataFrame,
    original_configuration: dict,
) -> dict:

    return {
        "version": "1.0",

        "dataset": {
            "name": "CIC-IDS2017",
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "ground_truth_column": GROUND_TRUTH_COLUMN,
            "label_column": LABEL_COLUMN,
            "attack_sessions": int(
                (df[GROUND_TRUTH_COLUMN] == 1).sum()
            ),
            "benign_sessions": int(
                (df[GROUND_TRUTH_COLUMN] == 0).sum()
            ),
        },

        "public_score": {
            "name": (
                "CIC-IDS2017 Public "
                "Network Behavioural Score"
            ),
            "type": (
                "external behavioural validation score"
            ),
            "score_range": [0, 100],
            "feature_count": len(PUBLIC_FEATURES),
            "features": PUBLIC_FEATURES,
            "weighting": (
                "equal_weight_percentile_normalization"
            ),
        },

        "score_validation": score_validation,

        "ground_truth_validation": {
            **statistical_validation,
            **classification_metrics,
        },

        "risk_band_distribution": (
            band_distribution.to_dict(
                orient="records"
            )
        ),

        "attack_rate_by_risk_band": (
            attack_rate_by_band.to_dict(
                orient="records"
            )
        ),

        "feature_effects": (
            feature_effects.to_dict(
                orient="records"
            )
        ),

        "original_model_feature_mapping": (
            mapping_report.to_dict(
                orient="records"
            )
        ),

        "limitations": [
            (
                "CIC-IDS2017 is primarily a network-flow "
                "dataset."
            ),
            (
                "SSH authentication success/failure is not "
                "directly represented."
            ),
            (
                "SSH usernames are not directly represented."
            ),
            (
                "HASSH is not directly represented."
            ),
            (
                "Shell commands are not directly represented."
            ),
            (
                "File operations are not directly represented."
            ),
            (
                "The public validation score is therefore "
                "not the original 12-feature risk score."
            ),
            (
                "The public score uses a separate network "
                "behavioural feature space."
            ),
            (
                "CIC-IDS2017 attack labels are external "
                "ground truth for this validation experiment, "
                "not proof of attacker intent in arbitrary "
                "real-world traffic."
            ),
        ],

        "research_interpretation": {
            "ground_truth_available": True,
            "external_dataset": True,
            "public_score_is_original_model": False,
            "attacker_classifier_claim": False,
            "attacker_intent_established": False,
            "deployment_ready": False,
            "interpretation": (
                "This experiment evaluates whether a separately "
                "constructed network behavioural score separates "
                "CIC-IDS2017 benign and labelled attack traffic. "
                "It does not establish attacker identity or intent "
                "and does not validate the original SSH/session "
                "risk model as directly applicable to CIC-IDS2017."
            ),
        },

        "original_risk_configuration_reference": {
            "version": original_configuration.get(
                "version"
            ),
            "method": original_configuration.get(
                "method"
            ),
            "selected_features": original_configuration.get(
                "selected_features"
            ),
        },
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print_header(
        "CIC-IDS2017 EXTERNAL BEHAVIOURAL RISK VALIDATION"
    )

    print("Input:")
    print(PUBLIC_DATASET)

    print()
    print("Output:")
    print(OUTPUT_DIR)

    # ------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------

    df = load_public_dataset()

    original_configuration = (
        load_original_configuration()
    )

    # ------------------------------------------------------------------
    # SCHEMA
    # ------------------------------------------------------------------

    validate_schema(df)

    # ------------------------------------------------------------------
    # CLEAN
    # ------------------------------------------------------------------

    df = clean_features(df)

    # ------------------------------------------------------------------
    # GROUND TRUTH
    # ------------------------------------------------------------------

    print_header("VALIDATING GROUND-TRUTH LABELS")

    unique_ground_truth = sorted(
        df[GROUND_TRUTH_COLUMN]
        .dropna()
        .unique()
        .tolist()
    )

    if not set(unique_ground_truth).issubset(
        {0, 1}
    ):
        raise ValueError(
            "is_attack must contain only binary "
            "0/1 values."
        )

    print(
        "Ground-truth values:",
        unique_ground_truth,
    )

    print()
    print(
        df[LABEL_COLUMN]
        .value_counts()
        .to_string()
    )

    # ------------------------------------------------------------------
    # FEATURE SCORE
    # ------------------------------------------------------------------

    feature_scores, feature_weights = (
        build_public_feature_scores(df)
    )

    # ------------------------------------------------------------------
    # RISK SCORE
    # ------------------------------------------------------------------

    public_score = calculate_public_risk_score(
        feature_scores,
        feature_weights,
    )

    df["public_risk_score"] = (
        public_score
    )

    df["risk_band"] = build_risk_bands(
        public_score
    )

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    score_validation = validate_scores(
        df,
        public_score,
    )

    # ------------------------------------------------------------------
    # DISTRIBUTION
    # ------------------------------------------------------------------

    print_header("BUILDING PUBLIC RISK-BAND DISTRIBUTION")

    band_distribution = (
        calculate_band_distribution(
            df["risk_band"]
        )
    )

    print(
        band_distribution.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # GROUND TRUTH DISTRIBUTION
    # ------------------------------------------------------------------

    ground_truth_distribution = (
        calculate_ground_truth_distribution(
            df
        )
    )

    # ------------------------------------------------------------------
    # SCORE BY GROUND TRUTH
    # ------------------------------------------------------------------

    score_by_ground_truth = (
        calculate_score_by_ground_truth(
            df
        )
    )

    print_header(
        "RISK SCORE BY GROUND-TRUTH CLASS"
    )

    print(
        score_by_ground_truth.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # SCORE BY ATTACK TYPE
    # ------------------------------------------------------------------

    score_by_label = (
        calculate_score_by_label(df)
    )

    print_header(
        "RISK SCORE BY CIC-IDS2017 LABEL"
    )

    print(
        score_by_label.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # ATTACK RATE BY BAND
    # ------------------------------------------------------------------

    attack_rate_by_band = (
        calculate_attack_rate_by_band(
            df
        )
    )

    print_header(
        "ATTACK RATE BY PUBLIC RISK BAND"
    )

    print(
        attack_rate_by_band.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------------

    statistical_validation = (
        calculate_statistical_validation(
            df
        )
    )

    # ------------------------------------------------------------------
    # ROC / PR
    # ------------------------------------------------------------------

    classification_metrics, roc_df, pr_df = (
        calculate_classification_metrics(
            df
        )
    )

    # ------------------------------------------------------------------
    # THRESHOLDS
    # ------------------------------------------------------------------

    threshold_metrics = (
        calculate_threshold_metrics(df)
    )

    if not threshold_metrics.empty:

        print_header(
            "THRESHOLD PERFORMANCE"
        )

        print(
            threshold_metrics.to_string(
                index=False
            )
        )

    # ------------------------------------------------------------------
    # FEATURE EFFECTS
    # ------------------------------------------------------------------

    feature_effects = (
        calculate_feature_ground_truth_effects(
            df
        )
    )

    # ------------------------------------------------------------------
    # ORIGINAL MODEL MAPPING
    # ------------------------------------------------------------------

    mapping_report = (
        build_feature_mapping_report()
    )

    # ------------------------------------------------------------------
    # SAVE SCORED DATASET
    # ------------------------------------------------------------------

    print_header(
        "SAVING PUBLIC RISK SCORE DATASET"
    )

    scored_columns = [
        ID_COLUMN,
        LABEL_COLUMN,
        GROUND_TRUTH_COLUMN,
        "public_risk_score",
        "risk_band",
    ]

    scored_columns.extend(
        feature_scores.columns.tolist()
    )

    scored_df = df[scored_columns].copy()

    scored_path = (
        OUTPUT_DIR
        / "public_risk_scores.csv"
    )

    scored_df.to_csv(
        scored_path,
        index=False,
    )

    print("Saved:")
    print(scored_path)

    # ------------------------------------------------------------------
    # SAVE DISTRIBUTIONS
    # ------------------------------------------------------------------

    ground_truth_path = (
        OUTPUT_DIR
        / "public_ground_truth_distribution.csv"
    )

    ground_truth_distribution.to_csv(
        ground_truth_path,
        index=False,
    )

    band_path = (
        OUTPUT_DIR
        / "public_risk_band_distribution.csv"
    )

    band_distribution.to_csv(
        band_path,
        index=False,
    )

    score_gt_path = (
        OUTPUT_DIR
        / "public_risk_by_ground_truth.csv"
    )

    score_by_ground_truth.to_csv(
        score_gt_path,
        index=False,
    )

    score_label_path = (
        OUTPUT_DIR
        / "public_risk_by_label.csv"
    )

    score_by_label.to_csv(
        score_label_path,
        index=False,
    )

    attack_band_path = (
        OUTPUT_DIR
        / "public_attack_rate_by_risk_band.csv"
    )

    attack_rate_by_band.to_csv(
        attack_band_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # SAVE ROC / PR
    # ------------------------------------------------------------------

    roc_path = (
        OUTPUT_DIR
        / "public_roc_curve.csv"
    )

    roc_df.to_csv(
        roc_path,
        index=False,
    )

    pr_path = (
        OUTPUT_DIR
        / "public_pr_curve.csv"
    )

    pr_df.to_csv(
        pr_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # SAVE THRESHOLD RESULTS
    # ------------------------------------------------------------------

    threshold_path = (
        OUTPUT_DIR
        / "public_threshold_metrics.csv"
    )

    threshold_metrics.to_csv(
        threshold_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # SAVE FEATURE EFFECTS
    # ------------------------------------------------------------------

    feature_effect_path = (
        OUTPUT_DIR
        / "public_feature_ground_truth_effects.csv"
    )

    feature_effects.to_csv(
        feature_effect_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # SAVE FEATURE MAPPING
    # ------------------------------------------------------------------

    mapping_path = (
        OUTPUT_DIR
        / "original_feature_mapping_review.csv"
    )

    mapping_report.to_csv(
        mapping_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # BUILD REPORT
    # ------------------------------------------------------------------

    print_header(
        "BUILDING PUBLIC VALIDATION REPORT"
    )

    report = build_validation_report(
        df=df,
        score_validation=score_validation,
        statistical_validation=(
            statistical_validation
        ),
        classification_metrics=(
            classification_metrics
        ),
        band_distribution=(
            band_distribution
        ),
        attack_rate_by_band=(
            attack_rate_by_band
        ),
        feature_effects=feature_effects,
        mapping_report=mapping_report,
        original_configuration=(
            original_configuration
        ),
    )

    # Add threshold metrics.
    report["threshold_metrics"] = (
        threshold_metrics.to_dict(
            orient="records"
        )
    )

    report_path = (
        OUTPUT_DIR
        / "public_risk_validation_report.json"
    )

    save_json(
        report_path,
        report,
    )

    print("Saved:")
    print(report_path)

    # ------------------------------------------------------------------
    # SAVE CONFIGURATION
    # ------------------------------------------------------------------

    public_configuration = {
        "version": "1.0",
        "dataset": "CIC-IDS2017",
        "score_name": (
            "CIC-IDS2017 Public "
            "Network Behavioural Score"
        ),
        "score_range": [0, 100],
        "normalization": (
            "empirical_percentile_rank"
        ),
        "weighting": (
            "equal_weight"
        ),
        "features": PUBLIC_FEATURES,
        "directions": PUBLIC_FEATURE_DIRECTIONS,
        "original_model_not_reused_directly": True,
        "original_model_configuration": (
            str(RISK_CONFIGURATION)
        ),
        "ground_truth": GROUND_TRUTH_COLUMN,
        "risk_bands": [
            {
                "name": name,
                "minimum": minimum,
                "maximum": maximum,
            }
            for name, minimum, maximum
            in RISK_BANDS
        ],
        "interpretation": {
            "external_validation": True,
            "attacker_classifier": False,
            "attacker_intent_established": False,
            "deployment_ready": False,
        },
    }

    configuration_path = (
        OUTPUT_DIR
        / "public_risk_validation_configuration.json"
    )

    save_json(
        configuration_path,
        public_configuration,
    )

    print("Saved:")
    print(configuration_path)

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------

    print_header(
        "CIC-IDS2017 EXTERNAL VALIDATION COMPLETE"
    )

    print(
        f"Rows validated: {len(df):,}"
    )

    print(
        f"Public score range: "
        f"{public_score.min():.4f} - "
        f"{public_score.max():.4f}"
    )

    print(
        f"Mean public score: "
        f"{public_score.mean():.4f}"
    )

    print(
        f"Median public score: "
        f"{public_score.median():.4f}"
    )

    if classification_metrics.get(
        "roc_auc"
    ) is not None:

        print(
            f"ROC-AUC: "
            f"{classification_metrics['roc_auc']:.6f}"
        )

        print(
            f"PR-AUC: "
            f"{classification_metrics['average_precision']:.6f}"
        )

    print()
    print(
        "IMPORTANT RESEARCH INTERPRETATION"
    )

    print(
        "1. CIC-IDS2017 provides independent ground-truth "
        "attack labels for this experiment."
    )

    print(
        "2. The public score is a separate network-flow "
        "behavioural validation score."
    )

    print(
        "3. It is NOT the original 12-feature SSH/session "
        "risk model."
    )

    print(
        "4. Missing SSH authentication, username, HASSH, "
        "command and file features were NOT fabricated."
    )

    print(
        "5. A good ROC-AUC or PR-AUC does NOT prove "
        "attacker intent."
    )

    print(
        "6. Do NOT deploy adaptive deception policies "
        "based on this validation alone."
    )

    print()
    print("Generated files:")

    generated_files = [
        scored_path,
        ground_truth_path,
        band_path,
        score_gt_path,
        score_label_path,
        attack_band_path,
        roc_path,
        pr_path,
        threshold_path,
        feature_effect_path,
        mapping_path,
        report_path,
        configuration_path,
    ]

    for index, path in enumerate(
        generated_files,
        start=1,
    ):
        print(f"{index}. {path}")


if __name__ == "__main__":
    main()