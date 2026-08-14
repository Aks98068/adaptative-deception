import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    Path("data")
    / "processed"
    / "session_features.json"
)

OUTPUT_FILE = (
    Path("data")
    / "processed"
    / "ml_features.csv"
)


# ============================================================
# FEATURES WE WANT TO KEEP
# ============================================================

FEATURE_COLUMNS = [

    # Temporal behaviour
    "duration_sec",
    "average_event_interval",
    "event_interval_variance",

    # General activity
    "event_count",
    "unique_event_types",

    # Command behaviour
    "num_commands",
    "command_entropy",

    # Authentication behaviour
    "failed_login_count",
    "successful_login_count",
    "failed_auth_ratio",

    # Identity / attacker fingerprint
    "unique_usernames",
    "unique_hassh",

    # File behaviour
    "num_file_events",

    # Session behaviour
    "session_closed_events",
]


# ============================================================
# LOAD JSON DATA
# ============================================================

def load_data():

    print()
    print("=" * 70)
    print("LOADING FEATURE DATA")
    print("=" * 70)

    print()

    print(
        f"Input file: {INPUT_FILE}"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file does not exist: "
            f"{INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    print(
        f"Records loaded: {len(data):,}"
    )

    return data


# ============================================================
# CREATE DATAFRAME
# ============================================================

def create_dataframe(data):

    print()
    print("=" * 70)
    print("CREATING DATAFRAME")
    print("=" * 70)

    print()

    df = pd.DataFrame(data)

    print(
        f"Rows:    {df.shape[0]:,}"
    )

    print(
        f"Columns: {df.shape[1]}"
    )

    return df


# ============================================================
# CHECK REQUIRED FEATURES
# ============================================================

def check_features(df):

    print()
    print("=" * 70)
    print("CHECKING REQUIRED FEATURES")
    print("=" * 70)

    print()

    missing_columns = [

        column
        for column in FEATURE_COLUMNS
        if column not in df.columns

    ]

    if missing_columns:

        print(
            "ERROR: Required features are missing:"
        )

        for column in missing_columns:

            print(
                f"  - {column}"
            )

        raise ValueError(
            "Required feature columns are missing."
        )

    print(
        "All required features are present."
    )


# ============================================================
# KEEP ONLY SELECTED FEATURES
# ============================================================

def select_features(df):

    print()
    print("=" * 70)
    print("SELECTING ML FEATURES")
    print("=" * 70)

    print()

    # Keep session ID separately.
    #
    # We need it for tracing predictions back
    # to the original session.
    selected = df[
        ["session_id"] +
        FEATURE_COLUMNS
    ].copy()

    print(
        f"Selected features: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print()

    for index, feature in enumerate(
        FEATURE_COLUMNS,
        start=1
    ):

        print(
            f"{index:2}. {feature}"
        )

    return selected


# ============================================================
# CONVERT NUMERICAL FEATURES
# ============================================================

def convert_numeric(df):

    print()
    print("=" * 70)
    print("CONVERTING FEATURES TO NUMERIC")
    print("=" * 70)

    print()

    for column in FEATURE_COLUMNS:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    print(
        "Numerical conversion complete."
    )

    return df


# ============================================================
# CHECK MISSING VALUES
# ============================================================

def check_missing_values(df):

    print()
    print("=" * 70)
    print("CHECKING MISSING VALUES")
    print("=" * 70)

    print()

    missing = (
        df[FEATURE_COLUMNS]
        .isnull()
        .sum()
    )

    total_missing = (
        missing.sum()
    )

    if total_missing == 0:

        print(
            "No missing feature values found."
        )

    else:

        print(
            "Missing values detected:"
        )

        print(
            missing[
                missing > 0
            ]
        )

    return df


# ============================================================
# CHECK INFINITE VALUES
# ============================================================

def check_infinite_values(df):

    print()
    print("=" * 70)
    print("CHECKING INFINITE VALUES")
    print("=" * 70)

    print()

    numerical_values = (
        df[FEATURE_COLUMNS]
        .to_numpy()
    )

    infinite_count = np.isinf(
        numerical_values
    ).sum()

    print(
        f"Infinite values: {infinite_count}"
    )

    if infinite_count > 0:

        print(
            "Replacing infinite values with NaN..."
        )

        df[FEATURE_COLUMNS] = (
            df[FEATURE_COLUMNS]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )

    return df


# ============================================================
# REMOVE DUPLICATE SESSIONS
# ============================================================

def remove_duplicate_sessions(df):

    print()
    print("=" * 70)
    print("CHECKING DUPLICATE SESSIONS")
    print("=" * 70)

    print()

    before = len(df)

    df = df.drop_duplicates(
        subset=["session_id"]
    )

    after = len(df)

    removed = before - after

    print(
        f"Duplicates removed: {removed:,}"
    )

    print(
        f"Remaining sessions: {after:,}"
    )

    return df


# ============================================================
# CHECK INVALID NUMBERS
# ============================================================

def check_invalid_values(df):

    print()
    print("=" * 70)
    print("CHECKING INVALID VALUES")
    print("=" * 70)

    print()

    # These features should never be negative.
    non_negative_features = [

        "duration_sec",
        "average_event_interval",
        "event_interval_variance",
        "event_count",
        "unique_event_types",
        "num_commands",
        "command_entropy",
        "failed_login_count",
        "successful_login_count",
        "failed_auth_ratio",
        "unique_usernames",
        "unique_hassh",
        "num_file_events",
        "session_closed_events",

    ]

    total_invalid = 0

    for column in non_negative_features:

        invalid_count = (
            df[column] < 0
        ).sum()

        if invalid_count > 0:

            print(
                f"{column}: "
                f"{invalid_count:,} "
                f"negative values"
            )

            total_invalid += (
                invalid_count
            )

    if total_invalid == 0:

        print(
            "No negative values found."
        )

    return df


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_summary(df):

    print()
    print("=" * 70)
    print("FINAL DATASET SUMMARY")
    print("=" * 70)

    print()

    print(
        f"Total sessions: "
        f"{len(df):,}"
    )

    print(
        f"Number of ML features: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print()

    print(
        "Feature matrix shape:"
    )

    print(
        f"({df.shape[0]:,}, "
        f"{len(FEATURE_COLUMNS)})"
    )

    print()

    print(
        "Features:"
    )

    for feature in FEATURE_COLUMNS:

        print(
            f"  - {feature}"
        )


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(df):

    print()
    print("=" * 70)
    print("SAVING ML DATASET")
    print("=" * 70)

    print()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Saved to:"
    )

    print(
        OUTPUT_FILE.resolve()
    )

    print()

    print(
        f"File size:"
    )

    size_mb = (
        OUTPUT_FILE.stat().st_size
        /
        (1024 * 1024)
    )

    print(
        f"{size_mb:.2f} MB"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ADAPTIVE DECEPTION AI")
    print("ML DATASET PREPARATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    data = load_data()

    # --------------------------------------------------------
    # 2. Create DataFrame
    # --------------------------------------------------------

    df = create_dataframe(
        data
    )

    # --------------------------------------------------------
    # 3. Check required features
    # --------------------------------------------------------

    check_features(
        df
    )

    # --------------------------------------------------------
    # 4. Select features
    # --------------------------------------------------------

    df = select_features(
        df
    )

    # --------------------------------------------------------
    # 5. Convert to numeric
    # --------------------------------------------------------

    df = convert_numeric(
        df
    )

    # --------------------------------------------------------
    # 6. Check missing values
    # --------------------------------------------------------

    df = check_missing_values(
        df
    )

    # --------------------------------------------------------
    # 7. Check infinite values
    # --------------------------------------------------------

    df = check_infinite_values(
        df
    )

    # --------------------------------------------------------
    # 8. Remove duplicate sessions
    # --------------------------------------------------------

    df = remove_duplicate_sessions(
        df
    )

    # --------------------------------------------------------
    # 9. Check invalid values
    # --------------------------------------------------------

    df = check_invalid_values(
        df
    )

    # --------------------------------------------------------
    # 10. Print summary
    # --------------------------------------------------------

    print_summary(
        df
    )

    # --------------------------------------------------------
    # 11. Save
    # --------------------------------------------------------

    save_dataset(
        df
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ML DATASET PREPARATION COMPLETE")
    print("=" * 70)

    print()
    print(
        "The dataset is now ready for the "
        "next research stage."
    )

    print()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()