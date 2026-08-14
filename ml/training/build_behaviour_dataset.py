from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/processed")

BASIC_FILE = DATA_DIR / "ml_features.csv"

SEQUENCE_FILE = DATA_DIR / "sequence_features.csv"

OUTPUT_FILE = DATA_DIR / "behavior_dataset.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print()
    print("=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)

    print()
    print("Loading basic ML features...")

    basic = pd.read_csv(BASIC_FILE)

    print(
        f"Basic dataset: {basic.shape}"
    )

    print()
    print("Loading sequence features...")

    sequence = pd.read_csv(SEQUENCE_FILE)

    print(
        f"Sequence dataset: {sequence.shape}"
    )

    return basic, sequence


# ============================================================
# PREPARE SEQUENCE FEATURES
# ============================================================

def prepare_sequence_features(sequence):

    columns = [

        "session_id",

        "event_count",

        "unique_event_types",

        "unique_event_transitions",

        "failed_login_count",

        "successful_login_count",

        "command_count",

        "file_event_count",

        "commands_after_login",

        "files_after_login",

        "time_to_failed_login_sec",

        "time_to_successful_login_sec",

        "time_to_first_command_sec",

        "time_to_first_file_event_sec",

        "session_stage"
    ]

    sequence = sequence[columns].copy()

    # --------------------------------------------------------
    # Rename sequence-specific columns
    # --------------------------------------------------------

    sequence = sequence.rename(
        columns={

            "event_count":
                "sequence_event_count",

            "unique_event_types":
                "sequence_unique_event_types",

            "failed_login_count":
                "sequence_failed_login_count",

            "successful_login_count":
                "sequence_successful_login_count"
        }
    )

    return sequence


# ============================================================
# ENCODE SESSION STAGE
# ============================================================

def encode_session_stage(df):

    mapping = {

        "connection": 0,

        "authentication_attempt": 1,

        "authenticated": 2,

        "command_activity": 3,

        "file_activity": 4
    }

    df["session_stage_encoded"] = (
        df["session_stage"]
        .map(mapping)
        .fillna(0)
        .astype("int8")
    )

    return df


# ============================================================
# CREATE BEHAVIOURAL FEATURES
# ============================================================

def create_behavior_features(df):

    print()
    print("=" * 70)
    print("CREATING BEHAVIOURAL FEATURES")
    print("=" * 70)

    # --------------------------------------------------------
    # Authentication intensity
    # --------------------------------------------------------

    df["authentication_intensity"] = (
        df["failed_login_count"]
        +
        df["successful_login_count"]
    )

    # --------------------------------------------------------
    # Post-authentication activity
    # --------------------------------------------------------

    df["post_auth_activity"] = (
        df["commands_after_login"]
        +
        df["files_after_login"]
    )

    # --------------------------------------------------------
    # Interactive activity
    # --------------------------------------------------------

    df["interactive_activity"] = (
        df["command_count"]
        +
        df["file_event_count"]
    )

    # --------------------------------------------------------
    # Transition density
    # --------------------------------------------------------

    df["transition_density"] = (
        df["unique_event_transitions"]
        /
        df["event_count"].clip(lower=1)
    )

    # --------------------------------------------------------
    # Successful authentication
    # --------------------------------------------------------

    df["has_successful_login"] = (
        df["successful_login_count"] > 0
    ).astype("int8")

    # --------------------------------------------------------
    # Command activity
    # --------------------------------------------------------

    df["has_command_activity"] = (
        df["command_count"] > 0
    ).astype("int8")

    # --------------------------------------------------------
    # File activity
    # --------------------------------------------------------

    df["has_file_activity"] = (
        df["file_event_count"] > 0
    ).astype("int8")

    # --------------------------------------------------------
    # Multi-stage behaviour
    # --------------------------------------------------------

    df["multi_stage_behavior"] = (
        (
            df["has_successful_login"]
            +
            df["has_command_activity"]
            +
            df["has_file_activity"]
        ) >= 2
    ).astype("int8")

    return df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df):

    print()
    print("=" * 70)
    print("CLEANING DATA")
    print("=" * 70)

    # Replace infinity
    df = df.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    # --------------------------------------------------------
    # Timing features
    # --------------------------------------------------------

    timing_columns = [

        "time_to_failed_login_sec",

        "time_to_successful_login_sec",

        "time_to_first_command_sec",

        "time_to_first_file_event_sec"
    ]

    for column in timing_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        df[column] = df[column].fillna(0)

    # --------------------------------------------------------
    # Convert all ML columns to numeric
    # --------------------------------------------------------

    for column in df.columns:

        if column != "session_id":

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Fill missing numerical values
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    df[numeric_columns] = (
        df[numeric_columns]
        .fillna(0)
    )

    return df


# ============================================================
# REMOVE NON-ML COLUMNS
# ============================================================

def remove_non_ml_columns(df):

    columns_to_remove = [

        "session_stage"
    ]

    return df.drop(
        columns=columns_to_remove,
        errors="ignore"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ADAPTIVE DECEPTION AI")
    print("BEHAVIOURAL DATASET CONSTRUCTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    basic, sequence = load_data()

    # --------------------------------------------------------
    # Prepare sequence data
    # --------------------------------------------------------

    sequence = prepare_sequence_features(
        sequence
    )

    # --------------------------------------------------------
    # Encode stage
    # --------------------------------------------------------

    sequence = encode_session_stage(
        sequence
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MERGING DATASETS")
    print("=" * 70)

    df = pd.merge(
        basic,
        sequence,
        on="session_id",
        how="inner"
    )

    print(
        f"Merged dataset shape: {df.shape}"
    )

    # --------------------------------------------------------
    # Create behavioural features
    # --------------------------------------------------------

    df = create_behavior_features(
        df
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    df = clean_data(
        df
    )

    # --------------------------------------------------------
    # Remove non-ML columns
    # --------------------------------------------------------

    df = remove_non_ml_columns(
        df
    )

    # --------------------------------------------------------
    # Remove duplicate sessions
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=["session_id"]
    )

    duplicates = before - len(df)

    print()
    print(
        f"Duplicate sessions removed: "
        f"{duplicates:,}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BEHAVIOURAL DATASET COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print()
    print("Final features:")

    for i, column in enumerate(
        df.columns,
        start=1
    ):

        print(
            f"{i:2}. {column}"
        )

    print()
    print("Saved to:")

    print(
        OUTPUT_FILE.resolve()
    )

    print()
    print("=" * 70)
    print("NEXT STEP: BEHAVIOURAL DATASET VALIDATION")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()