import json
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_FILE = (
    Path("data")
    / "processed"
    / "session_features.json"
)


# ============================================================
# LOAD FEATURE DATA
# ============================================================

def load_features():

    print("Loading feature dataset...")

    with open(
        FEATURE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    print(
        f"Records loaded: {len(data)}"
    )

    return data


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

def inspect_basic_information(df):

    print()
    print("=" * 70)
    print("1. BASIC DATASET INFORMATION")
    print("=" * 70)

    print()

    print(
        "Number of sessions:",
        len(df)
    )

    print(
        "Number of columns:",
        len(df.columns)
    )

    print()

    print("Columns:")

    for column in df.columns:

        print(
            f"  - {column}"
        )


# ============================================================
# DATA TYPES
# ============================================================

def inspect_data_types(df):

    print()
    print("=" * 70)
    print("2. DATA TYPES")
    print("=" * 70)

    print()

    print(
        df.dtypes
    )


# ============================================================
# MISSING VALUES
# ============================================================

def inspect_missing_values(df):

    print()
    print("=" * 70)
    print("3. MISSING VALUES")
    print("=" * 70)

    print()

    missing = df.isnull().sum()

    missing_percentage = (
        df.isnull().mean() * 100
    )

    result = pd.DataFrame({

        "missing_count":
            missing,

        "missing_percentage":
            missing_percentage

    })

    print(
        result
    )


# ============================================================
# NUMERICAL STATISTICS
# ============================================================

def inspect_statistics(df):

    print()
    print("=" * 70)
    print("4. NUMERICAL STATISTICS")
    print("=" * 70)

    print()

    numerical_df = df.select_dtypes(
        include="number"
    )

    print(
        numerical_df.describe().T
    )


# ============================================================
# ZERO VALUES
# ============================================================

def inspect_zero_values(df):

    print()
    print("=" * 70)
    print("5. ZERO VALUE ANALYSIS")
    print("=" * 70)

    print()

    numerical_df = df.select_dtypes(
        include="number"
    )

    for column in numerical_df.columns:

        zero_count = (
            numerical_df[column] == 0
        ).sum()

        percentage = (
            zero_count /
            len(df)
        ) * 100

        print(
            f"{column:35} "
            f"{zero_count:10} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# UNIQUE VALUES
# ============================================================

def inspect_unique_values(df):

    print()
    print("=" * 70)
    print("6. UNIQUE VALUE ANALYSIS")
    print("=" * 70)

    print()

    for column in df.columns:

        unique_count = (
            df[column].nunique()
        )

        print(
            f"{column:35} "
            f"{unique_count}"
        )


# ============================================================
# DUPLICATE SESSIONS
# ============================================================

def inspect_duplicates(df):

    print()
    print("=" * 70)
    print("7. DUPLICATE SESSION ANALYSIS")
    print("=" * 70)

    print()

    duplicate_count = (
        df["session_id"].duplicated().sum()
    )

    print(
        "Duplicate session IDs:",
        duplicate_count
    )


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def inspect_correlations(df):

    print()
    print("=" * 70)
    print("8. HIGH CORRELATION ANALYSIS")
    print("=" * 70)

    print()

    numerical_df = df.select_dtypes(
        include="number"
    )

    correlation = (
        numerical_df.corr()
    )

    columns = correlation.columns

    found = False

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            value = correlation.iloc[
                i,
                j
            ]

            if abs(value) >= 0.90:

                found = True

                print(
                    f"{columns[i]:35} "
                    f"<-> "
                    f"{columns[j]:35} "
                    f"{value:.3f}"
                )

    if not found:

        print(
            "No feature pairs with "
            "correlation >= 0.90 found."
        )


# ============================================================
# FEATURE VARIANCE
# ============================================================

def inspect_constant_features(df):

    print()
    print("=" * 70)
    print("9. CONSTANT / LOW-VARIANCE FEATURES")
    print("=" * 70)

    print()

    numerical_df = df.select_dtypes(
        include="number"
    )

    found = False

    for column in numerical_df.columns:

        unique_values = (
            numerical_df[column]
            .nunique()
        )

        if unique_values <= 1:

            found = True

            print(
                f"CONSTANT FEATURE: {column}"
            )

    if not found:

        print(
            "No completely constant "
            "numerical features found."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ADAPTIVE DECEPTION AI")
    print("DATA QUALITY INSPECTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    data = load_features()

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        data
    )

    # --------------------------------------------------------
    # Run inspections
    # --------------------------------------------------------

    inspect_basic_information(
        df
    )

    inspect_data_types(
        df
    )

    inspect_missing_values(
        df
    )

    inspect_statistics(
        df
    )

    inspect_zero_values(
        df
    )

    inspect_unique_values(
        df
    )

    inspect_duplicates(
        df
    )

    inspect_correlations(
        df
    )

    inspect_constant_features(
        df
    )

    print()
    print("=" * 70)
    print("DATA QUALITY INSPECTION COMPLETE")
    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()