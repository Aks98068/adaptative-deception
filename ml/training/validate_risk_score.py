"""
validate_risk_score.py

Research-stage validation of the transparent behavioural risk score.

IMPORTANT:
- This is NOT attacker classification.
- This does NOT establish attacker intent.
- This does NOT validate the score against ground-truth malicious labels.
- This stage evaluates distribution, consistency, contribution behaviour,
  and relationships with the previously generated unsupervised clusters.

Input:
    data/processed/risk/risk_scored_dataset.csv
    data/processed/risk/risk_score_configuration.json
    data/processed/clustering/results/clustered_sessions.csv

Output:
    data/processed/risk/validation/
        risk_score_distribution.csv
        risk_band_distribution.csv
        risk_score_percentiles.csv
        risk_contribution_summary.csv
        cluster_risk_summary.csv
        cluster_risk_band_distribution.csv
        risk_score_validation_report.json
"""

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

RISK_DIR = BASE_DIR / "data" / "processed" / "risk"
CLUSTER_DIR = BASE_DIR / "data" / "processed" / "clustering" / "results"

SCORED_FILE = RISK_DIR / "risk_scored_dataset.csv"
CONFIG_FILE = RISK_DIR / "risk_score_configuration.json"
CLUSTER_FILE = CLUSTER_DIR / "clustered_sessions.csv"

OUTPUT_DIR = RISK_DIR / "validation"


# ============================================================================
# EXPECTED FEATURES
# ============================================================================

EXPECTED_FEATURES = [
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

EXPECTED_BANDS = [
    "LOW",
    "GUARDED",
    "ELEVATED",
    "HIGH",
    "CRITICAL",
]


# ============================================================================
# UTILITIES
# ============================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def ensure_output_directory() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path, description: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )

    print(f"Loading:")
    print(path)

    df = pd.read_csv(path)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


def json_safe(value):
    """
    Convert NumPy/Pandas values into JSON-safe Python values.
    NaN and infinity become None.
    """

    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value

    if isinstance(value, np.ndarray):
        return [json_safe(x) for x in value.tolist()]

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [json_safe(v) for v in value]

    return value


def save_json(data: dict, path: Path) -> None:
    safe_data = json_safe(data)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            safe_data,
            f,
            indent=2,
            allow_nan=False,
        )


# ============================================================================
# VALIDATION
# ============================================================================

def validate_configuration(config: dict) -> None:
    print_header("VALIDATING RISK SCORE CONFIGURATION")

    selected_features = config.get("selected_features", [])

    if not selected_features:
        raise ValueError(
            "Configuration does not contain selected_features."
        )

    missing = [
        feature
        for feature in EXPECTED_FEATURES
        if feature not in selected_features
    ]

    if missing:
        raise ValueError(
            "Configuration is missing expected selected features:\n"
            + "\n".join(f" - {x}" for x in missing)
        )

    method = config.get("method", {})

    print(f"Scoring method: {method.get('name')}")
    print(f"Scoring type: {method.get('type')}")
    print(f"Normalization: {method.get('normalization')}")

    print()
    print("Selected features:")

    for i, feature in enumerate(selected_features, start=1):
        print(f"{i:2d}. {feature}")

    bands = config.get("risk_bands", [])

    if not bands:
        raise ValueError(
            "Configuration does not contain risk_bands."
        )

    print()
    print("Configured risk bands:")

    for band in bands:
        print(
            f"  {band.get('name'):10s} "
            f"{band.get('minimum')} - {band.get('maximum')}"
        )


def validate_scored_dataset(
    df: pd.DataFrame,
    config: dict,
) -> None:
    print_header("VALIDATING SCORED DATASET")

    required = [
        "session_id",
        "risk_score",
        "risk_band",
    ]

    for feature in EXPECTED_FEATURES:
        required.append(
            f"risk_contribution_{feature}"
        )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Required columns are missing:\n"
            + "\n".join(f" - {x}" for x in missing)
        )

    print("All required columns are present.")

    if df["session_id"].duplicated().any():
        duplicate_count = int(
            df["session_id"].duplicated().sum()
        )

        raise ValueError(
            f"Duplicate session IDs detected: {duplicate_count:,}"
        )

    if df["risk_score"].isna().any():
        raise ValueError(
            "Risk score contains missing values."
        )

    numeric_columns = [
        "risk_score"
    ]

    numeric_columns.extend(
        f"risk_contribution_{feature}"
        for feature in EXPECTED_FEATURES
    )

    for column in numeric_columns:
        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if values.isna().any():
            raise ValueError(
                f"Non-numeric values detected in {column}."
            )

        if np.isinf(values).any():
            raise ValueError(
                f"Infinite values detected in {column}."
            )

    minimum = float(df["risk_score"].min())
    maximum = float(df["risk_score"].max())

    if minimum < 0 or maximum > 100:
        raise ValueError(
            f"Risk scores outside expected 0-100 range: "
            f"{minimum} - {maximum}"
        )

    print(f"Rows: {len(df):,}")
    print(f"Risk score minimum: {minimum:.6f}")
    print(f"Risk score maximum: {maximum:.6f}")
    print("Missing values: 0")
    print("Infinite values: 0")
    print("Duplicate session IDs: 0")


# ============================================================================
# SCORE DISTRIBUTION
# ============================================================================

def build_score_distribution(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print_header("BUILDING RISK SCORE DISTRIBUTION")

    score = df["risk_score"]

    statistics = {
        "metric": [
            "count",
            "minimum",
            "maximum",
            "mean",
            "median",
            "standard_deviation",
            "variance",
            "skewness",
            "kurtosis",
        ],
        "value": [
            len(score),
            score.min(),
            score.max(),
            score.mean(),
            score.median(),
            score.std(),
            score.var(),
            score.skew(),
            score.kurt(),
        ],
    }

    result = pd.DataFrame(statistics)

    output = OUTPUT_DIR / "risk_score_distribution.csv"
    result.to_csv(output, index=False)

    print(f"Saved:\n{output}")

    return result


def build_percentiles(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print_header("CALCULATING RISK SCORE PERCENTILES")

    score = df["risk_score"]

    percentiles = [
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        0.995,
        0.999,
    ]

    rows = []

    for percentile in percentiles:
        rows.append(
            {
                "percentile": percentile * 100,
                "risk_score": score.quantile(percentile),
            }
        )

    result = pd.DataFrame(rows)

    output = OUTPUT_DIR / "risk_score_percentiles.csv"
    result.to_csv(output, index=False)

    print(result.to_string(index=False))
    print()
    print(f"Saved:\n{output}")

    return result


# ============================================================================
# RISK BANDS
# ============================================================================

def build_risk_band_distribution(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print_header("BUILDING RISK-BAND DISTRIBUTION")

    total = len(df)

    counts = (
        df["risk_band"]
        .value_counts()
        .reindex(EXPECTED_BANDS, fill_value=0)
    )

    rows = []

    for band in EXPECTED_BANDS:
        count = int(counts.loc[band])

        percentage = (
            count / total * 100
            if total > 0
            else 0
        )

        rows.append(
            {
                "risk_band": band,
                "sessions": count,
                "percentage": percentage,
            }
        )

    result = pd.DataFrame(rows)

    output = OUTPUT_DIR / "risk_band_distribution.csv"
    result.to_csv(output, index=False)

    print(result.to_string(index=False))
    print()
    print(f"Saved:\n{output}")

    return result


# ============================================================================
# CONTRIBUTIONS
# ============================================================================

def build_contribution_summary(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:

    print_header("VALIDATING FEATURE CONTRIBUTIONS")

    rows = []

    weights = config.get("normalized_weights", {})
    directions = config.get("feature_directions", {})

    for feature in EXPECTED_FEATURES:

        column = f"risk_contribution_{feature}"

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        absolute_values = values.abs()

        rows.append(
            {
                "feature": feature,
                "configured_weight": weights.get(
                    feature,
                    None,
                ),
                "configured_direction": directions.get(
                    feature,
                    None,
                ),
                "mean_contribution": values.mean(),
                "median_contribution": values.median(),
                "mean_absolute_contribution":
                    absolute_values.mean(),
                "maximum_absolute_contribution":
                    absolute_values.max(),
                "positive_percentage":
                    (values > 0).mean() * 100,
                "negative_percentage":
                    (values < 0).mean() * 100,
                "zero_percentage":
                    (values == 0).mean() * 100,
            }
        )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "mean_absolute_contribution",
        ascending=False,
    ).reset_index(drop=True)

    result.insert(
        0,
        "rank",
        range(1, len(result) + 1),
    )

    output = OUTPUT_DIR / "risk_contribution_summary.csv"
    result.to_csv(output, index=False)

    print(
        result[
            [
                "rank",
                "feature",
                "configured_weight",
                "mean_absolute_contribution",
                "positive_percentage",
                "negative_percentage",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Saved:\n{output}")

    return result


# ============================================================================
# CLUSTER COMPARISON
# ============================================================================

def build_cluster_comparison(
    risk_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    print_header("COMPARING RISK SCORES WITH BEHAVIOURAL CLUSTERS")

    if "session_id" not in cluster_df.columns:
        raise ValueError(
            "Cluster dataset does not contain session_id."
        )

    if "cluster" not in cluster_df.columns:
        raise ValueError(
            "Cluster dataset does not contain cluster."
        )

    cluster_subset = cluster_df[
        ["session_id", "cluster"]
    ].copy()

    cluster_subset = cluster_subset.drop_duplicates(
        subset=["session_id"]
    )

    merged = risk_df[
        ["session_id", "risk_score", "risk_band"]
    ].merge(
        cluster_subset,
        on="session_id",
        how="left",
        validate="one_to_one",
    )

    missing_cluster = int(
        merged["cluster"].isna().sum()
    )

    print(
        f"Sessions without cluster assignment: "
        f"{missing_cluster:,}"
    )

    merged = merged.dropna(
        subset=["cluster"]
    )

    merged["cluster"] = merged["cluster"].astype(int)

    summary_rows = []

    for cluster, group in merged.groupby("cluster"):

        scores = group["risk_score"]

        summary_rows.append(
            {
                "cluster": int(cluster),
                "sessions": len(group),
                "mean_risk_score": scores.mean(),
                "median_risk_score": scores.median(),
                "std_risk_score": scores.std(),
                "minimum_risk_score": scores.min(),
                "maximum_risk_score": scores.max(),
                "p25_risk_score": scores.quantile(0.25),
                "p75_risk_score": scores.quantile(0.75),
                "p95_risk_score": scores.quantile(0.95),
                "high_or_critical_percentage":
                    group["risk_band"]
                    .isin(["HIGH", "CRITICAL"])
                    .mean()
                    * 100,
                "critical_percentage":
                    (
                        group["risk_band"]
                        == "CRITICAL"
                    ).mean()
                    * 100,
            }
        )

    summary = pd.DataFrame(summary_rows)

    output_summary = (
        OUTPUT_DIR /
        "cluster_risk_summary.csv"
    )

    summary.to_csv(
        output_summary,
        index=False,
    )

    print()
    print(summary.to_string(index=False))
    print()
    print(f"Saved:\n{output_summary}")

    # ------------------------------------------------------------------------
    # Cluster x risk-band table
    # ------------------------------------------------------------------------

    band_table = pd.crosstab(
        merged["cluster"],
        merged["risk_band"],
    )

    band_table = band_table.reindex(
        columns=EXPECTED_BANDS,
        fill_value=0,
    )

    band_percentage = (
        band_table.div(
            band_table.sum(axis=1),
            axis=0,
        )
        * 100
    )

    rows = []

    for cluster in band_percentage.index:

        for band in EXPECTED_BANDS:

            rows.append(
                {
                    "cluster": int(cluster),
                    "risk_band": band,
                    "sessions": int(
                        band_table.loc[
                            cluster,
                            band,
                        ]
                    ),
                    "percentage": float(
                        band_percentage.loc[
                            cluster,
                            band,
                        ]
                    ),
                }
            )

    band_distribution = pd.DataFrame(rows)

    output_bands = (
        OUTPUT_DIR /
        "cluster_risk_band_distribution.csv"
    )

    band_distribution.to_csv(
        output_bands,
        index=False,
    )

    print()
    print(
        f"Saved:\n{output_bands}"
    )

    return summary, band_distribution


# ============================================================================
# FEATURE-SCORE RELATIONSHIP
# ============================================================================

def build_score_feature_correlations(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print_header("ANALYSING SCORE-CONTRIBUTION RELATIONSHIPS")

    rows = []

    for feature in EXPECTED_FEATURES:

        contribution_column = (
            f"risk_contribution_{feature}"
        )

        contribution = pd.to_numeric(
            df[contribution_column],
            errors="coerce",
        )

        correlation = df["risk_score"].corr(
            contribution,
            method="spearman",
        )

        rows.append(
            {
                "feature": feature,
                "spearman_correlation_with_risk_score":
                    correlation,
            }
        )

    result = pd.DataFrame(rows)

    result["absolute_correlation"] = (
        result[
            "spearman_correlation_with_risk_score"
        ]
        .abs()
    )

    result = result.sort_values(
        "absolute_correlation",
        ascending=False,
    )

    return result.reset_index(drop=True)


# ============================================================================
# SCORE STABILITY
# ============================================================================

def evaluate_score_stability(
    df: pd.DataFrame,
) -> dict:

    print_header("EVALUATING SCORE STABILITY")

    score = df["risk_score"]

    adjacent_changes = score.diff().abs()

    result = {
        "mean_adjacent_absolute_change":
            adjacent_changes.mean(),
        "median_adjacent_absolute_change":
            adjacent_changes.median(),
        "p95_adjacent_absolute_change":
            adjacent_changes.quantile(0.95),
        "maximum_adjacent_absolute_change":
            adjacent_changes.max(),
    }

    print(
        f"Mean adjacent change : "
        f"{result['mean_adjacent_absolute_change']:.6f}"
    )

    print(
        f"Median adjacent change : "
        f"{result['median_adjacent_absolute_change']:.6f}"
    )

    print(
        f"P95 adjacent change : "
        f"{result['p95_adjacent_absolute_change']:.6f}"
    )

    print(
        f"Maximum adjacent change : "
        f"{result['maximum_adjacent_absolute_change']:.6f}"
    )

    return result


# ============================================================================
# RESEARCH FLAGS
# ============================================================================

def build_research_flags(
    df: pd.DataFrame,
    band_distribution: pd.DataFrame,
    contribution_summary: pd.DataFrame,
    cluster_summary: pd.DataFrame,
) -> List[dict]:

    print_header("BUILDING RESEARCH VALIDATION FLAGS")

    flags = []

    # ------------------------------------------------------------------------
    # Flag 1: No ground truth
    # ------------------------------------------------------------------------

    flags.append(
        {
            "severity": "INFORMATION",
            "flag": "NO_GROUND_TRUTH",
            "message":
                "Risk score has not been validated against "
                "ground-truth attacker labels.",
        }
    )

    # ------------------------------------------------------------------------
    # Flag 2: Very concentrated risk bands
    # ------------------------------------------------------------------------

    elevated_high_critical = (
        band_distribution[
            band_distribution["risk_band"].isin(
                ["ELEVATED", "HIGH", "CRITICAL"]
            )
        ]["percentage"]
        .sum()
    )

    if elevated_high_critical > 80:
        flags.append(
            {
                "severity": "REVIEW",
                "flag": "HIGH_SCORE_CONCENTRATION",
                "message":
                    f"{elevated_high_critical:.2f}% of sessions "
                    "fall into ELEVATED/HIGH/CRITICAL bands.",
            }
        )

    # ------------------------------------------------------------------------
    # Flag 3: Sparse features
    # ------------------------------------------------------------------------

    sparse_features = []

    for feature in EXPECTED_FEATURES:

        column = f"risk_contribution_{feature}"

        zero_rate = (
            df[column] == 0
        ).mean()

        if zero_rate >= 0.95:
            sparse_features.append(
                {
                    "feature": feature,
                    "zero_rate": zero_rate * 100,
                }
            )

    if sparse_features:

        flags.append(
            {
                "severity": "REVIEW",
                "flag": "SPARSE_FEATURES",
                "message":
                    "Some contribution features are zero for "
                    "at least 95% of sessions.",
                "features": sparse_features,
            }
        )

    # ------------------------------------------------------------------------
    # Flag 4: Very small contributors
    # ------------------------------------------------------------------------

    weak = contribution_summary[
        contribution_summary[
            "mean_absolute_contribution"
        ]
        <
        contribution_summary[
            "mean_absolute_contribution"
        ].median()
        * 0.05
    ]

    if len(weak) > 0:

        flags.append(
            {
                "severity": "REVIEW",
                "flag": "WEAK_CONTRIBUTORS",
                "message":
                    "Some selected features contribute very little "
                    "to the final score.",
                "features":
                    weak["feature"].tolist(),
            }
        )

    # ------------------------------------------------------------------------
    # Flag 5: Cluster imbalance
    # ------------------------------------------------------------------------

    if len(cluster_summary) > 1:

        cluster_sizes = cluster_summary[
            "sessions"
        ]

        largest = cluster_sizes.max()
        smallest = cluster_sizes.min()

        if smallest > 0:

            imbalance_ratio = (
                largest / smallest
            )

            if imbalance_ratio > 10:

                flags.append(
                    {
                        "severity": "REVIEW",
                        "flag": "CLUSTER_SIZE_IMBALANCE",
                        "message":
                            "Behavioural clusters are highly "
                            "imbalanced.",
                        "largest_to_smallest_ratio":
                            imbalance_ratio,
                    }
                )

    print(f"Validation flags generated: {len(flags)}")

    for flag in flags:

        print(
            f"- {flag['severity']}: "
            f"{flag['flag']}"
        )

    return flags


# ============================================================================
# FINAL REPORT
# ============================================================================

def build_validation_report(
    df: pd.DataFrame,
    config: dict,
    distribution_df: pd.DataFrame,
    percentile_df: pd.DataFrame,
    band_df: pd.DataFrame,
    contribution_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    cluster_band_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
    stability: dict,
    flags: List[dict],
) -> dict:

    score = df["risk_score"]

    report = {
        "validation_version": "1.0",

        "dataset": {
            "rows": len(df),
            "columns": len(df.columns),
            "unique_sessions":
                df["session_id"].nunique(),
        },

        "risk_score": {
            "minimum": score.min(),
            "maximum": score.max(),
            "mean": score.mean(),
            "median": score.median(),
            "standard_deviation": score.std(),
        },

        "risk_bands": (
            band_df.to_dict(
                orient="records"
            )
        ),

        "percentiles": (
            percentile_df.to_dict(
                orient="records"
            )
        ),

        "feature_contributions": (
            contribution_df.to_dict(
                orient="records"
            )
        ),

        "feature_score_correlations": (
            correlation_df.to_dict(
                orient="records"
            )
        ),

        "cluster_comparison": (
            cluster_df.to_dict(
                orient="records"
            )
        ),

        "cluster_risk_bands": (
            cluster_band_df.to_dict(
                orient="records"
            )
        ),

        "stability": stability,

        "research_interpretation": {
            "ground_truth_available": False,
            "attacker_classifier": False,
            "attacker_intent_established": False,
            "deployment_ready": False,
            "interpretation":
                "The score represents behavioural separation "
                "derived from reviewed unsupervised features. "
                "It does not establish malicious intent or "
                "attacker identity.",
        },

        "validation_flags": flags,

        "configuration_reference": {
            "method": config.get(
                "method",
                {},
            ),
            "selected_features": config.get(
                "selected_features",
                [],
            ),
            "risk_bands": config.get(
                "risk_bands",
                [],
            ),
        },
    }

    return report


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print_header(
        "BEHAVIOURAL RISK SCORE VALIDATION"
    )

    print()
    print("Research validation stage")
    print("No attacker labels are assumed.")
    print("No deployment decision is made.")

    ensure_output_directory()

    # ------------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------------

    print_header("LOADING RISK SCORE CONFIGURATION")

    config = load_json(CONFIG_FILE)

    validate_configuration(config)

    # ------------------------------------------------------------------------
    # Load risk score dataset
    # ------------------------------------------------------------------------

    print_header("LOADING SCORED RISK DATASET")

    risk_df = load_csv(
        SCORED_FILE,
        "risk scored dataset",
    )

    # ------------------------------------------------------------------------
    # Validate risk dataset
    # ------------------------------------------------------------------------

    validate_scored_dataset(
        risk_df,
        config,
    )

    # ------------------------------------------------------------------------
    # Distribution
    # ------------------------------------------------------------------------

    distribution_df = build_score_distribution(
        risk_df
    )

    percentile_df = build_percentiles(
        risk_df
    )

    band_df = build_risk_band_distribution(
        risk_df
    )

    # ------------------------------------------------------------------------
    # Contributions
    # ------------------------------------------------------------------------

    contribution_df = build_contribution_summary(
        risk_df,
        config,
    )

    # ------------------------------------------------------------------------
    # Cluster comparison
    # ------------------------------------------------------------------------

    cluster_summary = pd.DataFrame()

    cluster_band_distribution = pd.DataFrame()

    if CLUSTER_FILE.exists():

        print_header(
            "LOADING BEHAVIOURAL CLUSTER DATASET"
        )

        cluster_df = load_csv(
            CLUSTER_FILE,
            "clustered behavioural dataset",
        )

        (
            cluster_summary,
            cluster_band_distribution,
        ) = build_cluster_comparison(
            risk_df,
            cluster_df,
        )

    else:

        print()
        print(
            "WARNING: Cluster dataset not found."
        )

        print(CLUSTER_FILE)

    # ------------------------------------------------------------------------
    # Feature correlations
    # ------------------------------------------------------------------------

    correlation_df = build_score_feature_correlations(
        risk_df
    )

    print()
    print(
        correlation_df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------------
    # Stability
    # ------------------------------------------------------------------------

    stability = evaluate_score_stability(
        risk_df
    )

    # ------------------------------------------------------------------------
    # Flags
    # ------------------------------------------------------------------------

    flags = build_research_flags(
        risk_df,
        band_df,
        contribution_df,
        cluster_summary,
    )

    # ------------------------------------------------------------------------
    # Save correlation results
    # ------------------------------------------------------------------------

    correlation_output = (
        OUTPUT_DIR /
        "risk_score_feature_correlations.csv"
    )

    correlation_df.to_csv(
        correlation_output,
        index=False,
    )

    print()
    print(
        f"Saved:\n{correlation_output}"
    )

    # ------------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------------

    print_header(
        "BUILDING RISK SCORE VALIDATION REPORT"
    )

    report = build_validation_report(
        risk_df,
        config,
        distribution_df,
        percentile_df,
        band_df,
        contribution_df,
        cluster_summary,
        cluster_band_distribution,
        correlation_df,
        stability,
        flags,
    )

    report_output = (
        OUTPUT_DIR /
        "risk_score_validation_report.json"
    )

    save_json(
        report,
        report_output,
    )

    print()
    print(
        f"Saved:\n{report_output}"
    )

    # ------------------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------------------

    print_header(
        "RISK SCORE VALIDATION COMPLETE"
    )

    print(
        f"Rows validated: {len(risk_df):,}"
    )

    print(
        f"Risk score range: "
        f"{risk_df['risk_score'].min():.4f} - "
        f"{risk_df['risk_score'].max():.4f}"
    )

    print(
        f"Mean risk score: "
        f"{risk_df['risk_score'].mean():.4f}"
    )

    print(
        f"Median risk score: "
        f"{risk_df['risk_score'].median():.4f}"
    )

    print()
    print("Generated files:")

    print(
        f"1. {OUTPUT_DIR / 'risk_score_distribution.csv'}"
    )

    print(
        f"2. {OUTPUT_DIR / 'risk_score_percentiles.csv'}"
    )

    print(
        f"3. {OUTPUT_DIR / 'risk_band_distribution.csv'}"
    )

    print(
        f"4. {OUTPUT_DIR / 'risk_contribution_summary.csv'}"
    )

    print(
        f"5. {OUTPUT_DIR / 'cluster_risk_summary.csv'}"
    )

    print(
        f"6. {OUTPUT_DIR / 'cluster_risk_band_distribution.csv'}"
    )

    print(
        f"7. {OUTPUT_DIR / 'risk_score_feature_correlations.csv'}"
    )

    print(
        f"8. {OUTPUT_DIR / 'risk_score_validation_report.json'}"
    )

    print()
    print("=" * 70)
    print("IMPORTANT RESEARCH CONSTRAINTS")
    print("=" * 70)

    print(
        "1. The score is an unsupervised behavioural score."
    )

    print(
        "2. The score is NOT an attacker classifier."
    )

    print(
        "3. The score does NOT establish attacker intent."
    )

    print(
        "4. Risk bands must NOT be interpreted as malicious labels."
    )

    print(
        "5. Adaptive deception policies should NOT be deployed yet."
    )

    print(
        "6. Ground-truth validation is still required before "
        "claiming detection performance."
    )


if __name__ == "__main__":
    main()