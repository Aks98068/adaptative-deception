import os
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/research_features.csv"

OUTPUT_DIR = "data/processed/clustering"

OUTPUT_DATASET = os.path.join(
    OUTPUT_DIR,
    "clustering_features.csv"
)

OUTPUT_METADATA = os.path.join(
    OUTPUT_DIR,
    "clustering_metadata.json"
)


# ============================================================
# FEATURES
# ============================================================

# Features we want to retain for behavioural profiling.
#
# We intentionally avoid several highly redundant features.

SELECTED_FEATURES = [
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
# LOG TRANSFORMATION
# ============================================================

LOG_FEATURES = [
    "duration_sec",
    "average_event_interval",
    "event_interval_variance",
    "event_count",
    "unique_event_types",
    "unique_event_transitions",
    "failed_login_count",
    "successful_login_count",
    "unique_usernames",
    "num_commands",
    "num_file_events",
    "time_to_failed_login_sec",
    "time_to_successful_login_sec",
    "time_to_first_command_sec",
    "time_to_first_file_event_sec",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    print("=" * 70)
    print("LOADING RESEARCH DATASET")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Could not find:\n{os.path.abspath(INPUT_FILE)}"
        )

    df = pd.read_csv(INPUT_FILE)

    print()
    print("Input shape:", df.shape)

    return df


# ============================================================
# CHECK REQUIRED FEATURES
# ============================================================

def validate_features(df):

    print()
    print("=" * 70)
    print("VALIDATING FEATURES")
    print("=" * 70)

    missing = [
        feature
        for feature in SELECTED_FEATURES
        if feature not in df.columns
    ]

    if missing:

        print()
        print("Missing features:")

        for feature in missing:
            print(" -", feature)

        raise ValueError(
            "Required clustering features are missing."
        )

    print()
    print("All clustering features are present.")


# ============================================================
# SELECT FEATURES
# ============================================================

def select_features(df):

    print()
    print("=" * 70)
    print("SELECTING BEHAVIOURAL FEATURES")
    print("=" * 70)

    result = df[
        ["session_id"] + SELECTED_FEATURES
    ].copy()

    print()
    print("Selected features:", len(SELECTED_FEATURES))

    for index, feature in enumerate(
        SELECTED_FEATURES,
        start=1
    ):
        print(f"{index:2d}. {feature}")

    return result


# ============================================================
# CLEAN NUMERIC VALUES
# ============================================================

def clean_numeric_values(df):

    print()
    print("=" * 70)
    print("CLEANING NUMERIC VALUES")
    print("=" * 70)

    feature_df = df[SELECTED_FEATURES].copy()

    for column in SELECTED_FEATURES:

        feature_df[column] = pd.to_numeric(
            feature_df[column],
            errors="coerce"
        )

    missing_before = int(
        feature_df.isna().sum().sum()
    )

    infinite_before = int(
        np.isinf(feature_df.to_numpy()).sum()
    )

    print()
    print("Missing values before cleaning:",
          missing_before)

    print(
        "Infinite values before cleaning:",
        infinite_before
    )

    # Replace infinite values with NaN
    feature_df = feature_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill missing numerical values with median
    for column in SELECTED_FEATURES:

        if feature_df[column].isna().any():

            median_value = feature_df[column].median()

            feature_df[column] = feature_df[column].fillna(
                median_value
            )

    print()
    print(
        "Missing values after cleaning:",
        int(feature_df.isna().sum().sum())
    )

    print(
        "Infinite values after cleaning:",
        int(np.isinf(feature_df.to_numpy()).sum())
    )

    return feature_df


# ============================================================
# LOG TRANSFORMATION
# ============================================================

def apply_log_transformation(df):

    print()
    print("=" * 70)
    print("APPLYING LOG TRANSFORMATIONS")
    print("=" * 70)

    transformed = df.copy()

    for column in LOG_FEATURES:

        if column not in transformed.columns:
            continue

        # log1p safely handles zero values
        transformed[column] = np.log1p(
            transformed[column].clip(lower=0)
        )

        print("Transformed:", column)

    return transformed


# ============================================================
# ROBUST SCALING
# ============================================================

def scale_features(df):

    print()
    print("=" * 70)
    print("ROBUST FEATURE SCALING")
    print("=" * 70)

    scaler = RobustScaler()

    scaled_array = scaler.fit_transform(
        df[SELECTED_FEATURES]
    )

    scaled_df = pd.DataFrame(
        scaled_array,
        columns=SELECTED_FEATURES
    )

    return scaled_df, scaler


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_final_dataset(df):

    print()
    print("=" * 70)
    print("FINAL DATASET VALIDATION")
    print("=" * 70)

    values = df[SELECTED_FEATURES].to_numpy()

    print()
    print("Rows:", len(df))

    print(
        "Features:",
        len(SELECTED_FEATURES)
    )

    print(
        "Missing values:",
        int(df[SELECTED_FEATURES].isna().sum().sum())
    )

    print(
        "Infinite values:",
        int(np.isinf(values).sum())
    )

    print()
    print("Final matrix shape:")

    print(
        values.shape
    )

    if df[SELECTED_FEATURES].isna().sum().sum() != 0:

        raise ValueError(
            "Dataset still contains missing values."
        )

    if np.isinf(values).sum() != 0:

        raise ValueError(
            "Dataset still contains infinite values."
        )


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(
    session_ids,
    scaled_df
):

    print()
    print("=" * 70)
    print("SAVING CLUSTERING DATASET")
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    final_df = pd.concat(
        [
            session_ids.reset_index(drop=True),
            scaled_df.reset_index(drop=True)
        ],
        axis=1
    )

    final_df.to_csv(
        OUTPUT_DATASET,
        index=False
    )

    print()
    print("Saved:")
    print(os.path.abspath(OUTPUT_DATASET))

    print()
    print(
        "Rows:",
        len(final_df)
    )

    print(
        "Columns:",
        len(final_df.columns)
    )

    return final_df


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata():

    metadata = {

        "purpose":
            "Unsupervised attacker behavioural profiling",

        "input_dataset":
            INPUT_FILE,

        "output_dataset":
            OUTPUT_DATASET,

        "scaling":
            "RobustScaler",

        "log_transformation":
            "log1p applied to highly skewed non-negative features",

        "feature_count":
            len(SELECTED_FEATURES),

        "features":
            SELECTED_FEATURES,

        "research_stage":
            "Pre-clustering behavioural representation",

        "next_stage":
            "Unsupervised behavioural clustering"
    }

    with open(
        OUTPUT_METADATA,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print()
    print("Metadata saved:")
    print(os.path.abspath(OUTPUT_METADATA))


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ADAPTIVE DECEPTION AI")
    print("CLUSTERING DATASET PREPARATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # 2. Validate
    # --------------------------------------------------------

    validate_features(df)

    # --------------------------------------------------------
    # 3. Select
    # --------------------------------------------------------

    selected_df = select_features(df)

    session_ids = selected_df[
        ["session_id"]
    ].copy()

    # --------------------------------------------------------
    # 4. Numeric cleaning
    # --------------------------------------------------------

    feature_df = clean_numeric_values(
        selected_df
    )

    # --------------------------------------------------------
    # 5. Log transformation
    # --------------------------------------------------------

    feature_df = apply_log_transformation(
        feature_df
    )

    # --------------------------------------------------------
    # 6. Scaling
    # --------------------------------------------------------

    scaled_df, scaler = scale_features(
        feature_df
    )

    # --------------------------------------------------------
    # 7. Validation
    # --------------------------------------------------------

    validate_final_dataset(
        scaled_df
    )

    # --------------------------------------------------------
    # 8. Save
    # --------------------------------------------------------

    save_dataset(
        session_ids,
        scaled_df
    )

    # --------------------------------------------------------
    # 9. Metadata
    # --------------------------------------------------------

    save_metadata()

    print()
    print("=" * 70)
    print("CLUSTERING DATASET PREPARATION COMPLETE")
    print("=" * 70)

    print()
    print("The dataset is now ready for:")
    print()
    print("1. Behavioural clustering")
    print("2. Attacker profiling")
    print("3. Cluster validation")
    print("4. Risk scoring")
    print("5. Adaptive deception policy learning")


if __name__ == "__main__":
    main()