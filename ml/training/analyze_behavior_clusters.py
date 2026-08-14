import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CLUSTER_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "clustering"
    / "results"
    / "clustered_sessions.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "clustering"
    / "results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

CLUSTER_COLUMN = "cluster"

# Features that are particularly important for behavioural
# interpretation.

BEHAVIOURAL_FEATURES = [
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


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("LOADING CLUSTERED DATASET")
    print("=" * 70)

    print(f"\nFile:")
    print(CLUSTER_FILE)

    df = pd.read_csv(CLUSTER_FILE)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    if CLUSTER_COLUMN not in df.columns:
        raise ValueError(
            f"Missing cluster column: {CLUSTER_COLUMN}"
        )

    print(
        f"\nClusters discovered: "
        f"{sorted(df[CLUSTER_COLUMN].unique())}"
    )

    return df


# ============================================================
# CLUSTER SIZE ANALYSIS
# ============================================================

def analyze_cluster_sizes(df):

    print("\n" + "=" * 70)
    print("1. CLUSTER SIZE ANALYSIS")
    print("=" * 70)

    counts = (
        df[CLUSTER_COLUMN]
        .value_counts()
        .sort_index()
    )

    percentages = (
        counts / len(df) * 100
    )

    result = pd.DataFrame(
        {
            "cluster": counts.index,
            "sessions": counts.values,
            "percentage": percentages.values,
        }
    )

    print()

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    return result


# ============================================================
# BEHAVIOURAL PROFILE
# ============================================================

def create_profile(df):

    print("\n" + "=" * 70)
    print("2. CLUSTER BEHAVIOURAL PROFILES")
    print("=" * 70)

    available_features = [
        f
        for f in BEHAVIOURAL_FEATURES
        if f in df.columns
    ]

    profile = (
        df.groupby(CLUSTER_COLUMN)[available_features]
        .mean()
        .T
    )

    print("\nMean feature values by cluster:")

    print(
        profile.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return profile


# ============================================================
# MEDIAN PROFILE
# ============================================================

def create_median_profile(df):

    print("\n" + "=" * 70)
    print("3. CLUSTER MEDIAN PROFILES")
    print("=" * 70)

    available_features = [
        f
        for f in BEHAVIOURAL_FEATURES
        if f in df.columns
    ]

    profile = (
        df.groupby(CLUSTER_COLUMN)[available_features]
        .median()
        .T
    )

    print("\nMedian feature values:")

    print(
        profile.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return profile


# ============================================================
# STANDARDIZED DIFFERENCE
# ============================================================

def calculate_effect_sizes(df):

    print("\n" + "=" * 70)
    print("4. CLUSTER DIFFERENCE ANALYSIS")
    print("=" * 70)

    available_features = [
        f
        for f in BEHAVIOURAL_FEATURES
        if f in df.columns
    ]

    clusters = sorted(
        df[CLUSTER_COLUMN].unique()
    )

    if len(clusters) != 2:

        print(
            "\nThis analysis currently expects exactly "
            "two clusters."
        )

        return None

    c0 = df[
        df[CLUSTER_COLUMN] == clusters[0]
    ]

    c1 = df[
        df[CLUSTER_COLUMN] == clusters[1]
    ]

    results = []

    for feature in available_features:

        mean0 = c0[feature].mean()
        mean1 = c1[feature].mean()

        median0 = c0[feature].median()
        median1 = c1[feature].median()

        std0 = c0[feature].std()
        std1 = c1[feature].std()

        pooled_std = np.sqrt(
            (
                std0 ** 2
                +
                std1 ** 2
            ) / 2
        )

        if pooled_std == 0:
            effect_size = 0
        else:
            effect_size = (
                mean1 - mean0
            ) / pooled_std

        results.append(
            {
                "feature": feature,
                "cluster_0_mean": mean0,
                "cluster_1_mean": mean1,
                "cluster_0_median": median0,
                "cluster_1_median": median1,
                "effect_size": effect_size,
                "absolute_effect_size": abs(effect_size),
            }
        )

    result = pd.DataFrame(results)

    result = result.sort_values(
        "absolute_effect_size",
        ascending=False
    )

    print(
        "\nFeatures ranked by behavioural difference:"
    )

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return result


# ============================================================
# BEHAVIOURAL INDICATORS
# ============================================================

def analyze_indicators(df):

    print("\n" + "=" * 70)
    print("5. BEHAVIOURAL INDICATOR ANALYSIS")
    print("=" * 70)

    indicators = {}

    if "successful_login_count" in df.columns:

        indicators[
            "successful_login"
        ] = df["successful_login_count"] > 0

    if "failed_login_count" in df.columns:

        indicators[
            "failed_login"
        ] = df["failed_login_count"] > 0

    if "num_commands" in df.columns:

        indicators[
            "command_activity"
        ] = df["num_commands"] > 0

    if "num_file_events" in df.columns:

        indicators[
            "file_activity"
        ] = df["num_file_events"] > 0

    if "duration_sec" in df.columns:

        indicators[
            "long_session"
        ] = df["duration_sec"] > 60

    rows = []

    for cluster in sorted(
        df[CLUSTER_COLUMN].unique()
    ):

        cluster_df = df[
            df[CLUSTER_COLUMN] == cluster
        ]

        row = {
            "cluster": cluster,
            "sessions": len(cluster_df),
        }

        for name, mask in indicators.items():

            cluster_mask = mask.loc[
                cluster_df.index
            ]

            row[
                f"{name}_percentage"
            ] = (
                cluster_mask.mean() * 100
            )

        rows.append(row)

    result = pd.DataFrame(rows)

    print()

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    return result


# ============================================================
# AUTOMATED INTERPRETATION
# ============================================================

def generate_interpretation(
    effect_results,
    indicator_results
):

    print("\n" + "=" * 70)
    print("6. INITIAL BEHAVIOURAL INTERPRETATION")
    print("=" * 70)

    if effect_results is not None:

        print(
            "\nTop behavioural separating features:"
        )

        top = effect_results.head(10)

        for _, row in top.iterrows():

            direction = (
                "higher in Cluster 1"
                if row["effect_size"] > 0
                else
                "higher in Cluster 0"
            )

            print(
                f"- {row['feature']}: "
                f"{direction} "
                f"(effect={row['effect_size']:.4f})"
            )

    print(
        "\nIMPORTANT:"
    )

    print(
        "These are statistical differences, not attacker labels."
    )

    print(
        "Human interpretation is required before assigning "
        "behavioural profile names."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    # --------------------------------------------------------
    # Cluster sizes
    # --------------------------------------------------------

    size_result = analyze_cluster_sizes(
        df
    )

    size_result.to_csv(
        OUTPUT_DIR
        / "cluster_sizes.csv",
        index=False
    )

    # --------------------------------------------------------
    # Mean profiles
    # --------------------------------------------------------

    mean_profile = create_profile(
        df
    )

    mean_profile.to_csv(
        OUTPUT_DIR
        / "cluster_mean_profiles.csv"
    )

    # --------------------------------------------------------
    # Median profiles
    # --------------------------------------------------------

    median_profile = create_median_profile(
        df
    )

    median_profile.to_csv(
        OUTPUT_DIR
        / "cluster_median_profiles.csv"
    )

    # --------------------------------------------------------
    # Effect sizes
    # --------------------------------------------------------

    effect_results = calculate_effect_sizes(
        df
    )

    if effect_results is not None:

        effect_results.to_csv(
            OUTPUT_DIR
            / "cluster_feature_differences.csv",
            index=False
        )

    # --------------------------------------------------------
    # Behavioural indicators
    # --------------------------------------------------------

    indicator_results = analyze_indicators(
        df
    )

    indicator_results.to_csv(
        OUTPUT_DIR
        / "cluster_behavior_indicators.csv",
        index=False
    )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    generate_interpretation(
        effect_results,
        indicator_results
    )

    print("\n" + "=" * 70)
    print("CLUSTER ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nNEXT:"
    )

    print(
        "Review the feature differences before "
        "assigning attacker profile names."
    )


if __name__ == "__main__":
    main()