"""
validate_cluster_interpretation.py

Research-stage statistical validation of behavioural clusters.

Purpose:
    1. Validate cluster-level feature differences.
    2. Calculate effect sizes.
    3. Detect redundant features.
    4. Produce a research validation report.

Input:
    data/processed/clustering/results/clustered_sessions.csv

Outputs:
    data/processed/clustering/results/
        cluster_statistical_tests.csv
        cluster_effect_sizes.csv
        cluster_feature_redundancy.csv
        cluster_validation_report.json
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "clustering"
    / "results"
)

INPUT_FILE = RESULTS_DIR / "clustered_sessions.csv"

STATISTICAL_FILE = RESULTS_DIR / "cluster_statistical_tests.csv"
EFFECT_FILE = RESULTS_DIR / "cluster_effect_sizes.csv"
REDUNDANCY_FILE = RESULTS_DIR / "cluster_feature_redundancy.csv"
REPORT_FILE = RESULTS_DIR / "cluster_validation_report.json"


FEATURES = [
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
    "num_file_events",
    "time_to_failed_login_sec",
    "time_to_successful_login_sec",
    "time_to_first_command_sec",
    "time_to_first_file_event_sec",
    "session_stage_encoded",
]


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def cliffs_delta(x, y):
    """
    Approximate Cliff's delta using rank statistics.

    Delta:
        +1 = x generally larger than y
         0 = similar distributions
        -1 = x generally smaller than y
    """

    x = np.asarray(x)
    y = np.asarray(y)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    combined = np.concatenate([x, y])

    ranks = pd.Series(combined).rank(method="average").to_numpy()

    rank_x = ranks[: len(x)]

    u = np.sum(rank_x) - (len(x) * (len(x) + 1)) / 2

    return (
        2 * u / (len(x) * len(y))
    ) - 1


def interpret_effect(delta):
    if pd.isna(delta):
        return "unknown"

    magnitude = abs(delta)

    if magnitude < 0.147:
        return "negligible"
    elif magnitude < 0.33:
        return "small"
    elif magnitude < 0.474:
        return "medium"
    else:
        return "large"


def safe_mannwhitney(x, y):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            statistic, p_value = mannwhitneyu(
                x,
                y,
                alternative="two-sided"
            )

        return statistic, p_value

    except Exception:
        return np.nan, np.nan


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    print_header("LOADING CLUSTERED DATASET")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{INPUT_FILE}"
        )

    print(f"Loading:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


# ============================================================
# VALIDATE STRUCTURE
# ============================================================

def validate_dataset(df):

    print_header("VALIDATING DATASET")

    required = ["session_id", "cluster"] + FEATURES

    missing = [
        feature
        for feature in required
        if feature not in df.columns
    ]

    if missing:

        print("Missing columns:")

        for feature in missing:
            print(f" - {feature}")

        raise ValueError(
            "Required validation features are missing."
        )

    print("All required validation features are present.")

    clusters = sorted(df["cluster"].dropna().unique())

    print(f"Clusters detected: {clusters}")

    if len(clusters) != 2:
        raise ValueError(
            f"This validation script expects exactly 2 clusters. "
            f"Detected: {clusters}"
        )

    return clusters


# ============================================================
# STATISTICAL TESTS
# ============================================================

def statistical_tests(df, clusters):

    print_header("STATISTICAL CLUSTER COMPARISON")

    cluster_a = clusters[0]
    cluster_b = clusters[1]

    df_a = df[df["cluster"] == cluster_a]
    df_b = df[df["cluster"] == cluster_b]

    results = []

    for feature in FEATURES:

        x = pd.to_numeric(
            df_a[feature],
            errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        y = pd.to_numeric(
            df_b[feature],
            errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if len(x) == 0 or len(y) == 0:
            continue

        statistic, p_value = safe_mannwhitney(
            x,
            y
        )

        delta = cliffs_delta(
            x,
            y
        )

        results.append({
            "feature": feature,
            "cluster_a": cluster_a,
            "cluster_b": cluster_b,
            "cluster_a_n": len(x),
            "cluster_b_n": len(y),
            "cluster_a_median": x.median(),
            "cluster_b_median": y.median(),
            "cluster_a_mean": x.mean(),
            "cluster_b_mean": y.mean(),
            "mann_whitney_u": statistic,
            "p_value": p_value,
            "cliffs_delta": delta,
            "effect_magnitude": interpret_effect(delta),
        })

    result_df = pd.DataFrame(results)

    # Bonferroni correction
    if len(result_df) > 0:

        result_df["p_value_adjusted"] = np.minimum(
            result_df["p_value"] * len(result_df),
            1.0
        )

        result_df["statistically_significant"] = (
            result_df["p_value_adjusted"] < 0.05
        )

    result_df = result_df.sort_values(
        "cliffs_delta",
        key=lambda x: x.abs(),
        ascending=False
    )

    result_df.to_csv(
        STATISTICAL_FILE,
        index=False
    )

    print(
        f"Statistical results saved:\n{STATISTICAL_FILE}"
    )

    return result_df


# ============================================================
# EFFECT SIZE ANALYSIS
# ============================================================

def effect_size_analysis(stat_df):

    print_header("EFFECT SIZE ANALYSIS")

    effects = stat_df[
        [
            "feature",
            "cliffs_delta",
            "effect_magnitude",
            "p_value_adjusted",
            "statistically_significant",
        ]
    ].copy()

    effects["absolute_effect"] = (
        effects["cliffs_delta"].abs()
    )

    effects = effects.sort_values(
        "absolute_effect",
        ascending=False
    )

    effects.to_csv(
        EFFECT_FILE,
        index=False
    )

    print(
        effects.head(20).to_string(index=False)
    )

    print()
    print(f"Saved:\n{EFFECT_FILE}")

    return effects


# ============================================================
# REDUNDANCY ANALYSIS
# ============================================================

def redundancy_analysis(df):

    print_header("FEATURE REDUNDANCY ANALYSIS")

    correlation = df[FEATURES].corr(
        method="spearman"
    )

    rows = []

    for i in range(len(FEATURES)):

        for j in range(i + 1, len(FEATURES)):

            feature_a = FEATURES[i]
            feature_b = FEATURES[j]

            corr = correlation.loc[
                feature_a,
                feature_b
            ]

            if abs(corr) >= 0.90:

                rows.append({
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "spearman_correlation": corr,
                    "absolute_correlation": abs(corr),
                })

    redundancy_df = pd.DataFrame(rows)

    if not redundancy_df.empty:

        redundancy_df = redundancy_df.sort_values(
            "absolute_correlation",
            ascending=False
        )

    redundancy_df.to_csv(
        REDUNDANCY_FILE,
        index=False
    )

    print(
        f"High-correlation pairs: {len(redundancy_df)}"
    )

    if not redundancy_df.empty:
        print(
            redundancy_df.head(30).to_string(
                index=False
            )
        )

    print()
    print(f"Saved:\n{REDUNDANCY_FILE}")

    return redundancy_df


# ============================================================
# CLUSTER SIZE VALIDATION
# ============================================================

def cluster_size_analysis(df, clusters):

    print_header("CLUSTER SIZE ANALYSIS")

    counts = (
        df["cluster"]
        .value_counts()
        .sort_index()
    )

    percentages = (
        counts / len(df) * 100
    )

    result = []

    for cluster in clusters:

        result.append({
            "cluster": int(cluster),
            "sessions": int(counts.get(cluster, 0)),
            "percentage": float(
                percentages.get(cluster, 0)
            ),
        })

    result_df = pd.DataFrame(result)

    print(
        result_df.to_string(index=False)
    )

    return result_df


# ============================================================
# BUILD REPORT
# ============================================================

def build_report(
    df,
    clusters,
    stat_df,
    effect_df,
    redundancy_df,
    cluster_sizes,
):

    print_header("BUILDING VALIDATION REPORT")

    significant = stat_df[
        stat_df["statistically_significant"]
    ]

    large_effects = effect_df[
        effect_df["absolute_effect"] >= 0.474
    ]

    report = {

        "dataset": {
            "rows": int(len(df)),
            "features_tested": len(FEATURES),
        },

        "clusters": [
            int(x)
            for x in clusters
        ],

        "cluster_sizes":
            cluster_sizes.to_dict(
                orient="records"
            ),

        "statistical_testing": {
            "method":
                "Mann-Whitney U test",
            "multiple_testing_correction":
                "Bonferroni",
            "significance_threshold":
                0.05,
            "significant_features":
                int(len(significant)),
        },

        "effect_size": {
            "method":
                "Cliff's delta",
            "large_effect_features":
                int(len(large_effects)),
        },

        "redundancy": {
            "method":
                "Spearman correlation",
            "threshold":
                0.90,
            "high_correlation_pairs":
                int(len(redundancy_df)),
        },

        "top_separating_features":
            effect_df.head(10)[
                [
                    "feature",
                    "cliffs_delta",
                    "effect_magnitude",
                    "statistically_significant",
                ]
            ].to_dict(
                orient="records"
            ),

        "research_interpretation": {

            "cluster_labels":
                "No attacker labels assigned.",

            "warning":
                "Statistical cluster differences do not establish attacker intent.",

            "next_stage":
                "Feature selection and risk-scoring design.",
        },
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    print(
        f"Validation report saved:\n{REPORT_FILE}"
    )

    return report


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CLUSTER INTERPRETATION VALIDATION")
    print("=" * 70)

    df = load_dataset()

    clusters = validate_dataset(df)

    stat_df = statistical_tests(
        df,
        clusters
    )

    effect_df = effect_size_analysis(
        stat_df
    )

    redundancy_df = redundancy_analysis(
        df
    )

    cluster_sizes = cluster_size_analysis(
        df,
        clusters
    )

    build_report(
        df,
        clusters,
        stat_df,
        effect_df,
        redundancy_df,
        cluster_sizes,
    )

    print()
    print("=" * 70)
    print("CLUSTER VALIDATION COMPLETE")
    print("=" * 70)

    print()
    print("Generated:")
    print(f"1. {STATISTICAL_FILE}")
    print(f"2. {EFFECT_FILE}")
    print(f"3. {REDUNDANCY_FILE}")
    print(f"4. {REPORT_FILE}")

    print()
    print("IMPORTANT:")
    print(
        "Statistical significance does not establish attacker intent."
    )
    print(
        "Review the effect sizes and redundant features before risk scoring."
    )


if __name__ == "__main__":
    main()