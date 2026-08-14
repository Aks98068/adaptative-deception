"""
Interpret K-Means behavioural clusters.

Purpose:
    Analyse the statistically identified clusters and generate
    evidence-based behavioural profiles.

Important:
    This script DOES NOT assign "attacker" or "legitimate" labels.
    It only describes behavioural differences between clusters.

Input:
    data/processed/clustering/results/clustered_sessions.csv

Outputs:
    data/processed/clustering/results/
        cluster_profiles.csv
        cluster_feature_effects.csv
        cluster_interpretation.json
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
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

PROFILES_FILE = OUTPUT_DIR / "cluster_profiles.csv"
EFFECTS_FILE = OUTPUT_DIR / "cluster_feature_effects.csv"
JSON_FILE = OUTPUT_DIR / "cluster_interpretation.json"


# ============================================================
# FEATURES USED FOR INTERPRETATION
# ============================================================
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
# HELPER FUNCTIONS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def safe_percentage(numerator, denominator):
    if denominator == 0:
        return 0.0

    return (numerator / denominator) * 100.0


def safe_mean(series):
    if len(series) == 0:
        return 0.0

    value = series.mean()

    if pd.isna(value):
        return 0.0

    return float(value)


# ============================================================
# LOAD DATA
# ============================================================

def load_clustered_dataset():

    print_header("LOADING CLUSTERED DATASET")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find clustered dataset:\n{INPUT_FILE}"
        )

    print(f"Loading:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    if "cluster" not in df.columns:
        raise ValueError(
            "The clustered dataset does not contain a 'cluster' column."
        )

    return df


# ============================================================
# VALIDATE FEATURES
# ============================================================

def validate_features(df):

    print_header("VALIDATING INTERPRETATION FEATURES")

    missing = []

    for feature in FEATURES:

        if feature not in df.columns:
            missing.append(feature)

    if missing:

        print("Missing features:")

        for feature in missing:
            print(f" - {feature}")

        raise ValueError(
            "Required interpretation features are missing."
        )

    print("All interpretation features are present.")


# ============================================================
# BASIC CLUSTER SUMMARY
# ============================================================

def create_cluster_summary(df):

    print_header("CREATING CLUSTER SUMMARY")

    clusters = sorted(df["cluster"].dropna().unique())

    total_sessions = len(df)

    rows = []

    for cluster in clusters:

        cluster_df = df[df["cluster"] == cluster]

        sessions = len(cluster_df)

        percentage = safe_percentage(
            sessions,
            total_sessions
        )

        successful_login_percentage = safe_percentage(
            (cluster_df["successful_login_count"] > 0).sum(),
            sessions
        )

        failed_login_percentage = safe_percentage(
            (cluster_df["failed_login_count"] > 0).sum(),
            sessions
        )

        command_percentage = safe_percentage(
            (cluster_df["num_commands"] > 0).sum(),
            sessions
        )

        file_percentage = safe_percentage(
            (cluster_df["num_file_events"] > 0).sum(),
            sessions
        )

        long_session_percentage = safe_percentage(
            (cluster_df["duration_sec"] > 60).sum(),
            sessions
        )

        rows.append(
            {
                "cluster": int(cluster),
                "sessions": int(sessions),
                "percentage": round(percentage, 4),
                "successful_login_percentage": round(
                    successful_login_percentage,
                    4
                ),
                "failed_login_percentage": round(
                    failed_login_percentage,
                    4
                ),
                "command_activity_percentage": round(
                    command_percentage,
                    4
                ),
                "file_activity_percentage": round(
                    file_percentage,
                    4
                ),
                "long_session_percentage": round(
                    long_session_percentage,
                    4
                ),
            }
        )

    summary = pd.DataFrame(rows)

    print(summary.to_string(index=False))

    return summary


# ============================================================
# FEATURE STATISTICS BY CLUSTER
# ============================================================

def calculate_cluster_statistics(df):

    print_header("CALCULATING CLUSTER FEATURE STATISTICS")

    clusters = sorted(df["cluster"].dropna().unique())

    rows = []

    for feature in FEATURES:

        for cluster in clusters:

            values = df.loc[
                df["cluster"] == cluster,
                feature
            ]

            rows.append(
                {
                    "feature": feature,
                    "cluster": int(cluster),
                    "mean": safe_mean(values),
                    "median": float(values.median()),
                    "std": float(values.std()),
                }
            )

    statistics = pd.DataFrame(rows)

    return statistics


# ============================================================
# EFFECT SIZE
# ============================================================

def calculate_effect_sizes(df):

    print_header("CALCULATING FEATURE EFFECT SIZES")

    clusters = sorted(df["cluster"].dropna().unique())

    if len(clusters) != 2:

        print(
            "WARNING: Effect-size interpretation currently "
            "supports exactly 2 clusters."
        )

        return pd.DataFrame()

    cluster_a = clusters[0]
    cluster_b = clusters[1]

    df_a = df[df["cluster"] == cluster_a]
    df_b = df[df["cluster"] == cluster_b]

    rows = []

    for feature in FEATURES:

        mean_a = safe_mean(df_a[feature])
        mean_b = safe_mean(df_b[feature])

        std_a = float(df_a[feature].std())
        std_b = float(df_b[feature].std())

        pooled_std = np.sqrt(
            (
                (std_a ** 2)
                +
                (std_b ** 2)
            ) / 2
        )

        if pooled_std == 0 or np.isnan(pooled_std):

            effect = 0.0

        else:

            effect = (
                mean_b - mean_a
            ) / pooled_std

        rows.append(
            {
                "feature": feature,
                f"cluster_{cluster_a}_mean": mean_a,
                f"cluster_{cluster_b}_mean": mean_b,
                "effect_cluster_b_minus_a": effect,
                "absolute_effect": abs(effect),
            }
        )

    effects = pd.DataFrame(rows)

    effects = effects.sort_values(
        "absolute_effect",
        ascending=False
    )

    return effects


# ============================================================
# FEATURE INTERPRETATION
# ============================================================

def interpret_effect(effect):

    absolute_effect = abs(effect)

    if absolute_effect >= 2.0:
        return "very_strong"

    if absolute_effect >= 1.0:
        return "strong"

    if absolute_effect >= 0.5:
        return "moderate"

    if absolute_effect >= 0.2:
        return "small"

    return "weak"


def create_feature_interpretations(effects):

    print_header("TOP BEHAVIOURAL DIFFERENCES")

    if effects.empty:
        return []

    interpretations = []

    for _, row in effects.head(15).iterrows():

        feature = row["feature"]

        effect = float(
            row["effect_cluster_b_minus_a"]
        )

        strength = interpret_effect(effect)

        if effect > 0:

            higher_cluster = "Cluster 1"
            lower_cluster = "Cluster 0"

        elif effect < 0:

            higher_cluster = "Cluster 0"
            lower_cluster = "Cluster 1"

        else:

            higher_cluster = "Neither"
            lower_cluster = "Neither"

        interpretation = {
            "feature": feature,
            "effect": round(effect, 4),
            "absolute_effect": round(abs(effect), 4),
            "strength": strength,
            "higher_cluster": higher_cluster,
            "lower_cluster": lower_cluster,
        }

        interpretations.append(interpretation)

        print(
            f"{feature:40s} "
            f"effect={effect:8.4f} "
            f"strength={strength:12s} "
            f"higher={higher_cluster}"
        )

    return interpretations


# ============================================================
# BEHAVIOURAL PROFILE GENERATION
# ============================================================

def generate_profile_names(summary):

    print_header("GENERATING PROVISIONAL BEHAVIOURAL PROFILES")

    profiles = []

    for _, row in summary.iterrows():

        cluster = int(row["cluster"])

        successful = row[
            "successful_login_percentage"
        ]

        failed = row[
            "failed_login_percentage"
        ]

        commands = row[
            "command_activity_percentage"
        ]

        files = row[
            "file_activity_percentage"
        ]

        # IMPORTANT:
        # These are behavioural descriptions only.
        # They are NOT attacker/legitimate labels.

        if successful >= 50 and failed < 10:

            profile_name = (
                "Successful Authentication / "
                "Interactive Behaviour"
            )

            rationale = (
                "High proportion of sessions with successful "
                "authentication and relatively low failed "
                "authentication activity."
            )

        elif failed > successful:

            profile_name = (
                "Authentication-Failure Dominant Behaviour"
            )

            rationale = (
                "Failed authentication activity is more prevalent "
                "than successful authentication activity."
            )

        else:

            profile_name = (
                "Mixed Authentication Behaviour"
            )

            rationale = (
                "The cluster contains a mixture of successful "
                "and unsuccessful authentication behaviour."
            )

        profiles.append(
            {
                "cluster": cluster,
                "provisional_profile": profile_name,
                "rationale": rationale,
                "successful_login_percentage": successful,
                "failed_login_percentage": failed,
                "command_activity_percentage": commands,
                "file_activity_percentage": files,
            }
        )

    return profiles


# ============================================================
# CONFIDENCE ESTIMATION
# ============================================================

def estimate_profile_confidence(summary, effects):

    print_header("ESTIMATING INTERPRETATION CONFIDENCE")

    if len(summary) != 2:

        return {
            "level": "low",
            "reason": (
                "Automated confidence estimation requires "
                "exactly two clusters."
            ),
        }

    top_effects = effects.head(5)

    strong_count = (
        top_effects["absolute_effect"] >= 1.0
    ).sum()

    if strong_count >= 3:

        level = "high"

    elif strong_count >= 1:

        level = "moderate"

    else:

        level = "low"

    reason = (
        f"{strong_count} of the five strongest behavioural "
        f"features have an absolute effect size >= 1.0."
    )

    print(f"Confidence: {level}")
    print(f"Reason: {reason}")

    return {
        "level": level,
        "reason": reason,
    }


# ============================================================
# SAVE JSON REPORT
# ============================================================

def save_interpretation(
    summary,
    effects,
    profiles,
    feature_interpretations,
    confidence
):

    print_header("SAVING INTERPRETATION REPORT")

    cluster_data = []

    for profile in profiles:

        cluster = profile["cluster"]

        cluster_effects = []

        for item in feature_interpretations:

            cluster_effects.append(
                item
            )

        cluster_data.append(
            {
                "cluster": cluster,
                "profile": profile,
                "top_behavioural_features": cluster_effects,
            }
        )

    report = {
        "research_note": (
            "Cluster labels are behavioural descriptions only. "
            "They do not represent ground-truth attacker or "
            "legitimate-user labels."
        ),

        "total_sessions": int(
            summary["sessions"].sum()
        ),

        "number_of_clusters": int(
            len(summary)
        ),

        "interpretation_confidence": confidence,

        "clusters": cluster_data,
    }

    with open(
        JSON_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    print(f"Saved:\n{JSON_FILE}")


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "BEHAVIOURAL CLUSTER INTERPRETATION"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. LOAD
    # --------------------------------------------------------

    df = load_clustered_dataset()

    # --------------------------------------------------------
    # 2. VALIDATE
    # --------------------------------------------------------

    validate_features(df)

    # --------------------------------------------------------
    # 3. SUMMARY
    # --------------------------------------------------------

    summary = create_cluster_summary(df)

    # --------------------------------------------------------
    # 4. STATISTICS
    # --------------------------------------------------------

    statistics = calculate_cluster_statistics(df)

    # --------------------------------------------------------
    # 5. EFFECT SIZES
    # --------------------------------------------------------

    effects = calculate_effect_sizes(df)

    # --------------------------------------------------------
    # 6. INTERPRET FEATURES
    # --------------------------------------------------------

    feature_interpretations = (
        create_feature_interpretations(
            effects
        )
    )

    # --------------------------------------------------------
    # 7. PROFILES
    # --------------------------------------------------------

    profiles = generate_profile_names(
        summary
    )

    # --------------------------------------------------------
    # 8. CONFIDENCE
    # --------------------------------------------------------

    confidence = estimate_profile_confidence(
        summary,
        effects
    )

    # --------------------------------------------------------
    # 9. SAVE PROFILE CSV
    # --------------------------------------------------------

    profile_df = pd.DataFrame(
        profiles
    )

    profile_df.to_csv(
        PROFILES_FILE,
        index=False
    )

    # --------------------------------------------------------
    # 10. SAVE EFFECT CSV
    # --------------------------------------------------------

    effects.to_csv(
        EFFECTS_FILE,
        index=False
    )

    # --------------------------------------------------------
    # 11. SAVE JSON
    # --------------------------------------------------------

    save_interpretation(
        summary,
        effects,
        profiles,
        feature_interpretations,
        confidence
    )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print_header(
        "CLUSTER INTERPRETATION COMPLETE"
    )

    print(
        f"Profiles saved:\n{PROFILES_FILE}"
    )

    print(
        f"\nFeature effects saved:\n{EFFECTS_FILE}"
    )

    print(
        f"\nInterpretation report saved:\n{JSON_FILE}"
    )

    print()
    print("IMPORTANT:")
    print(
        "The generated profile names are provisional "
        "behavioural descriptions."
    )

    print(
        "Do NOT treat them as ground-truth attacker labels."
    )

    print()
    print(
        "NEXT STEP:"
    )

    print(
        "Review the cluster profiles before building "
        "the risk-scoring layer."
    )


if __name__ == "__main__":
    main()