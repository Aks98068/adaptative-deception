from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATH
# ============================================================

DATA_FILE = Path(
    "data/processed/behavior_dataset.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    print("=" * 70)
    print("LOADING BEHAVIOURAL DATASET")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)

    print()
    print("Dataset shape:")
    print(df.shape)

    return df


# ============================================================
# BASIC INFORMATION
# ============================================================

def inspect_basic_information(df):

    print()
    print("=" * 70)
    print("1. BASIC INFORMATION")
    print("=" * 70)

    print()

    print("Rows:")
    print(f"{len(df):,}")

    print()

    print("Columns:")
    print(len(df.columns))

    print()

    print("Memory usage:")
    print(
        f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
    )


# ============================================================
# MISSING VALUES
# ============================================================

def inspect_missing_values(df):

    print()
    print("=" * 70)
    print("2. MISSING VALUES")
    print("=" * 70)

    missing = df.isnull().sum()

    missing_percentage = (
        missing / len(df)
    ) * 100

    result = pd.DataFrame({

        "missing_count": missing,

        "missing_percentage":
            missing_percentage

    })

    result = result[
        result["missing_count"] > 0
    ]

    if result.empty:

        print()
        print("No missing values found.")

    else:

        print(result)


# ============================================================
# INFINITE VALUES
# ============================================================

def inspect_infinite_values(df):

    print()
    print("=" * 70)
    print("3. INFINITE VALUES")
    print("=" * 70)

    numeric = df.select_dtypes(
        include=np.number
    )

    infinite = np.isinf(
        numeric
    ).sum()

    infinite = infinite[
        infinite > 0
    ]

    if infinite.empty:

        print()
        print("No infinite values found.")

    else:

        print(infinite)


# ============================================================
# DUPLICATE SESSIONS
# ============================================================

def inspect_duplicates(df):

    print()
    print("=" * 70)
    print("4. DUPLICATE SESSION ANALYSIS")
    print("=" * 70)

    duplicates = df[
        "session_id"
    ].duplicated().sum()

    print()
    print(
        f"Duplicate sessions: {duplicates:,}"
    )


# ============================================================
# CONSTANT FEATURES
# ============================================================

def inspect_constant_features(df):

    print()
    print("=" * 70)
    print("5. CONSTANT / LOW-VARIANCE FEATURES")
    print("=" * 70)

    numeric = df.select_dtypes(
        include=np.number
    )

    unique_counts = numeric.nunique()

    constant = unique_counts[
        unique_counts <= 1
    ]

    if constant.empty:

        print()
        print("No constant features.")

    else:

        print()
        print("Constant features:")

        for column in constant.index:

            print(
                f" - {column}"
            )


# ============================================================
# CORRELATION
# ============================================================

def inspect_correlations(df):

    print()
    print("=" * 70)
    print("6. HIGH CORRELATION ANALYSIS")
    print("=" * 70)

    numeric = df.select_dtypes(
        include=np.number
    )

    correlation = numeric.corr()

    pairs = []

    columns = correlation.columns

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            value = correlation.iloc[i, j]

            if abs(value) >= 0.90:

                pairs.append({

                    "feature_1":
                        columns[i],

                    "feature_2":
                        columns[j],

                    "correlation":
                        round(value, 4)

                })

    result = pd.DataFrame(
        pairs
    )

    if result.empty:

        print()
        print(
            "No correlation >= 0.90 found."
        )

    else:

        result = result.sort_values(
            "correlation",
            key=lambda x: abs(x),
            ascending=False
        )

        print()

        print(result.to_string(
            index=False
        ))


# ============================================================
# BEHAVIOURAL SUMMARY
# ============================================================

def behavioural_summary(df):

    print()
    print("=" * 70)
    print("7. BEHAVIOURAL SUMMARY")
    print("=" * 70)

    total = len(df)

    commands = (
        df["has_command_activity"]
        .sum()
    )

    files = (
        df["has_file_activity"]
        .sum()
    )

    successful = (
        df["has_successful_login"]
        .sum()
    )

    multi_stage = (
        df["multi_stage_behavior"]
        .sum()
    )

    long_sessions = (
        df["duration_sec"] > 60
    ).sum()

    print()

    print(
        f"Total sessions: {total:,}"
    )

    print(
        f"Sessions with commands: "
        f"{commands:,} "
        f"({commands / total * 100:.2f}%)"
    )

    print(
        f"Sessions with file activity: "
        f"{files:,} "
        f"({files / total * 100:.2f}%)"
    )

    print(
        f"Sessions with successful login: "
        f"{successful:,} "
        f"({successful / total * 100:.2f}%)"
    )

    print(
        f"Multi-stage behaviour: "
        f"{multi_stage:,} "
        f"({multi_stage / total * 100:.2f}%)"
    )

    print(
        f"Sessions longer than 60 seconds: "
        f"{long_sessions:,} "
        f"({long_sessions / total * 100:.2f}%)"
    )


# ============================================================
# FEATURE STATISTICS
# ============================================================

def inspect_statistics(df):

    print()
    print("=" * 70)
    print("8. FEATURE STATISTICS")
    print("=" * 70)

    numeric = df.select_dtypes(
        include=np.number
    )

    statistics = numeric.describe().T

    statistics[
        "missing"
    ] = numeric.isnull().sum()

    statistics[
        "zero_count"
    ] = (numeric == 0).sum()

    print()

    print(
        statistics.to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ADAPTIVE DECEPTION AI")
    print("BEHAVIOURAL DATASET VALIDATION")
    print("=" * 70)

    df = load_dataset()

    inspect_basic_information(
        df
    )

    inspect_missing_values(
        df
    )

    inspect_infinite_values(
        df
    )

    inspect_duplicates(
        df
    )

    inspect_constant_features(
        df
    )

    inspect_correlations(
        df
    )

    behavioural_summary(
        df
    )

    inspect_statistics(
        df
    )

    print()
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    print()
    print(
        "DO NOT TRAIN THE MODEL YET."
    )

    print(
        "Review the validation results first."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()