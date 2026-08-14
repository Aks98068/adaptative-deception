"""
prepare_risk_dataset.py

Purpose
-------
Prepare the reviewed behavioural features for the future risk-scoring layer.

This script DOES NOT:
    - assign attacker labels
    - train a classifier
    - claim attacker intent
    - create ground-truth labels
    - perform adaptive deception

It only:
    1. Loads the reviewed risk feature list.
    2. Loads the behavioural dataset.
    3. Extracts the 12 selected features.
    4. Validates the data.
    5. Applies safe transformations to highly skewed count/time features.
    6. Robust-scales the features.
    7. Preserves session IDs and raw feature values.
    8. Produces a risk-scoring preparation dataset.
    9. Saves metadata for reproducibility.

Input
-----
data/processed/behavior_dataset.csv
data/processed/risk/selected_risk_features.csv
data/processed/risk/risk_feature_review.csv

Output
------
data/processed/risk/risk_dataset.csv
data/processed/risk/risk_scaling_metadata.json
data/processed/risk/risk_dataset_report.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler


# ======================================================================
# PATHS
# ======================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
RISK_DIR = DATA_DIR / "risk"

BEHAVIOUR_FILE = DATA_DIR / "behavior_dataset.csv"
SELECTED_FEATURE_FILE = RISK_DIR / "selected_risk_features.csv"
REVIEW_FILE = RISK_DIR / "risk_feature_review.csv"

OUTPUT_DATASET = RISK_DIR / "risk_dataset.csv"
OUTPUT_SCALING = RISK_DIR / "risk_scaling_metadata.json"
OUTPUT_REPORT = RISK_DIR / "risk_dataset_report.json"


# ======================================================================
# EXPECTED FEATURES FROM YOUR REVIEWED FEATURE SELECTION
# ======================================================================

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


# ======================================================================
# FEATURE TRANSFORMATIONS
# ======================================================================

# Highly right-skewed variables are log-transformed using:
#
#     log1p(x) = log(1 + x)
#
# This is only for the modelling representation.
# Raw values are preserved separately.

LOG_TRANSFORM_FEATURES = [
    "event_interval_variance",
    "duration_sec",
    "time_to_failed_login_sec",
    "successful_login_count",
    "unique_event_types",
    "unique_event_transitions",
    "unique_hassh",
    "unique_usernames",
    "num_file_events",
    "time_to_first_command_sec",
]


# ======================================================================
# HELPERS
# ======================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def ensure_directory() -> None:
    RISK_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path, description: str) -> pd.DataFrame:
    print(f"Loading {description}...")
    print(path)

    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file does not exist:\n{path}\n"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"\nFile is empty:\n{path}\n"
        )

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


def make_json_safe(value):
    """
    Convert numpy/pandas values into JSON-safe Python values.

    NaN and infinity are converted to None.
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

    if isinstance(value, np.ndarray):
        return [
            make_json_safe(v)
            for v in value.tolist()
        ]

    if isinstance(value, dict):
        return {
            str(k): make_json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            make_json_safe(v)
            for v in value
        ]

    return value


def save_json(path: Path, data: dict) -> None:
    safe_data = make_json_safe(data)

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            safe_data,
            file,
            indent=4,
            allow_nan=False,
        )


# ======================================================================
# SELECTED FEATURE DISCOVERY
# ======================================================================

def extract_selected_features(
    selected_df: pd.DataFrame,
) -> List[str]:

    print_header("EXTRACTING REVIEWED RISK FEATURES")

    # --------------------------------------------------------------
    # Case 1:
    # The selection file is a feature matrix:
    #
    # session_id, feature1, feature2, ...
    #
    # This is the format your current selected_risk_features.csv uses.
    # --------------------------------------------------------------

    matrix_features = [
        column
        for column in selected_df.columns
        if column != "session_id"
    ]

    if set(EXPECTED_FEATURES).issubset(set(matrix_features)):

        selected = [
            feature
            for feature in EXPECTED_FEATURES
            if feature in matrix_features
        ]

        print("Detected feature-matrix format.")

    # --------------------------------------------------------------
    # Case 2:
    # The selection file is a long feature-selection table:
    #
    # feature, effect_size, ...
    # --------------------------------------------------------------

    elif "feature" in selected_df.columns:

        selected = (
            selected_df["feature"]
            .dropna()
            .astype(str)
            .tolist()
        )

        selected = [
            feature
            for feature in selected
            if feature in EXPECTED_FEATURES
        ]

        print("Detected feature-list format.")

    else:

        raise ValueError(
            "\nCould not determine the format of "
            "selected_risk_features.csv.\n\n"
            "Expected either:\n"
            "  1. A feature matrix containing the selected columns, or\n"
            "  2. A table containing a 'feature' column.\n"
        )

    # Remove duplicates while preserving order.
    selected = list(dict.fromkeys(selected))

    if not selected:
        raise ValueError(
            "No recognised risk features were found."
        )

    print()
    print(f"Selected features found: {len(selected)}")

    for index, feature in enumerate(selected, start=1):
        print(f"{index:2d}. {feature}")

    # --------------------------------------------------------------
    # Safety check.
    #
    # We expect exactly the reviewed 12-feature set.
    # --------------------------------------------------------------

    missing_expected = [
        feature
        for feature in EXPECTED_FEATURES
        if feature not in selected
    ]

    unexpected = [
        feature
        for feature in selected
        if feature not in EXPECTED_FEATURES
    ]

    if missing_expected:
        raise ValueError(
            "\nThe reviewed feature set is incomplete.\n"
            f"Missing features: {missing_expected}"
        )

    if unexpected:
        raise ValueError(
            "\nUnexpected features detected:\n"
            f"{unexpected}"
        )

    return EXPECTED_FEATURES.copy()


# ======================================================================
# DATA VALIDATION
# ======================================================================

def validate_input_features(
    df: pd.DataFrame,
    selected_features: List[str],
) -> None:

    print_header("VALIDATING INPUT FEATURES")

    required = ["session_id"] + selected_features

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "\nMissing required columns:\n"
            + "\n".join(f" - {x}" for x in missing)
        )

    print("All required columns are present.")

    # --------------------------------------------------------------
    # Numeric validation
    # --------------------------------------------------------------

    for feature in selected_features:

        if not pd.api.types.is_numeric_dtype(df[feature]):

            print(
                f"Converting non-numeric feature: {feature}"
            )

            df[feature] = pd.to_numeric(
                df[feature],
                errors="coerce",
            )

    print("Numeric validation complete.")


def clean_numeric_values(
    df: pd.DataFrame,
    selected_features: List[str],
) -> pd.DataFrame:

    print_header("CLEANING RISK FEATURES")

    result = df.copy()

    # Replace infinities.
    result[selected_features] = (
        result[selected_features]
        .replace([np.inf, -np.inf], np.nan)
    )

    missing_before = int(
        result[selected_features]
        .isna()
        .sum()
        .sum()
    )

    print(f"Missing/infinite values before cleaning: {missing_before:,}")

    # Median imputation.
    #
    # This should normally have very little effect because your
    # research dataset was already validated with zero missing values.
    #

    imputation_values = {}

    for feature in selected_features:

        missing = result[feature].isna()

        if missing.any():

            median_value = result[feature].median()

            if pd.isna(median_value):
                raise ValueError(
                    f"Feature '{feature}' contains no valid numeric values."
                )

            result.loc[missing, feature] = median_value

            imputation_values[feature] = float(
                median_value
            )

    missing_after = int(
        result[selected_features]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Missing/infinite values after cleaning: "
        f"{missing_after:,}"
    )

    if missing_after != 0:
        raise ValueError(
            "Missing values remain after cleaning."
        )

    return result


# ======================================================================
# RAW FEATURE COPY
# ======================================================================

def create_raw_feature_columns(
    df: pd.DataFrame,
    selected_features: List[str],
) -> pd.DataFrame:

    print_header("PRESERVING RAW FEATURES")

    result = df.copy()

    for feature in selected_features:

        raw_name = f"raw_{feature}"

        result[raw_name] = result[feature].astype(float)

    print(
        f"Preserved {len(selected_features)} raw feature columns."
    )

    return result


# ======================================================================
# TRANSFORMATIONS
# ======================================================================

def transform_features(
    df: pd.DataFrame,
    selected_features: List[str],
):
    print_header("APPLYING FEATURE TRANSFORMATIONS")

    transformed = pd.DataFrame(
        index=df.index
    )

    transformation_metadata = {}

    for feature in selected_features:

        series = df[feature].astype(float)

        if feature in LOG_TRANSFORM_FEATURES:

            # Ensure non-negative values.
            minimum = float(series.min())

            if minimum < 0:

                shift = abs(minimum)

                transformed_values = np.log1p(
                    series + shift
                )

                transformation_metadata[feature] = {
                    "transformation": "log1p_shifted",
                    "shift": shift,
                }

            else:

                transformed_values = np.log1p(
                    series
                )

                transformation_metadata[feature] = {
                    "transformation": "log1p",
                    "shift": 0.0,
                }

        else:

            transformed_values = series

            transformation_metadata[feature] = {
                "transformation": "none",
                "shift": 0.0,
            }

        transformed[feature] = transformed_values

        print(
            f"Transformed: {feature}"
            if feature in LOG_TRANSFORM_FEATURES
            else f"Preserved: {feature}"
        )

    return transformed, transformation_metadata


# ======================================================================
# ROBUST SCALING
# ======================================================================

def robust_scale_features(
    transformed_df: pd.DataFrame,
    selected_features: List[str],
):
    print_header("ROBUST FEATURE SCALING")

    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
    )

    matrix = scaler.fit_transform(
        transformed_df[selected_features]
    )

    scaled_df = pd.DataFrame(
        matrix,
        columns=[
            f"risk_scaled_{feature}"
            for feature in selected_features
        ],
        index=transformed_df.index,
    )

    # --------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------

    scaling_metadata = {
        "scaler": "RobustScaler",
        "with_centering": True,
        "with_scaling": True,
        "quantile_range": [25.0, 75.0],
        "features": {},
    }

    for index, feature in enumerate(selected_features):

        scaling_metadata["features"][feature] = {
            "center": float(
                scaler.center_[index]
            ),
            "scale": float(
                scaler.scale_[index]
            ),
        }

    print(
        f"Scaled features: {len(selected_features)}"
    )

    return scaled_df, scaling_metadata


# ======================================================================
# DATASET VALIDATION
# ======================================================================

def validate_final_dataset(
    df: pd.DataFrame,
    selected_features: List[str],
    scaled_features: List[str],
) -> dict:

    print_header("FINAL RISK DATASET VALIDATION")

    numeric_columns = (
        selected_features +
        scaled_features
    )

    missing = int(
        df[numeric_columns]
        .isna()
        .sum()
        .sum()
    )

    infinite = int(
        np.isinf(
            df[numeric_columns]
            .to_numpy(
                dtype=float
            )
        ).sum()
    )

    print(f"Rows: {len(df):,}")
    print(
        f"Raw features: {len(selected_features)}"
    )
    print(
        f"Scaled features: {len(scaled_features)}"
    )
    print(
        f"Missing values: {missing:,}"
    )
    print(
        f"Infinite values: {infinite:,}"
    )

    if missing != 0:
        raise ValueError(
            "Final dataset contains missing values."
        )

    if infinite != 0:
        raise ValueError(
            "Final dataset contains infinite values."
        )

    # --------------------------------------------------------------
    # Session ID validation
    # --------------------------------------------------------------

    duplicate_sessions = int(
        df["session_id"].duplicated().sum()
    )

    print(
        f"Duplicate session IDs: "
        f"{duplicate_sessions:,}"
    )

    # Session IDs should normally be unique because this is a
    # session-level risk dataset.

    if duplicate_sessions != 0:

        raise ValueError(
            "Duplicate session IDs detected."
        )

    # --------------------------------------------------------------
    # Scaled-feature statistics
    # --------------------------------------------------------------

    scaled_stats = {}

    for feature in scaled_features:

        values = df[feature]

        scaled_stats[feature] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(values.median()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    return {
        "rows": int(len(df)),
        "raw_feature_count": len(selected_features),
        "scaled_feature_count": len(scaled_features),
        "missing_values": missing,
        "infinite_values": infinite,
        "duplicate_session_ids": duplicate_sessions,
        "scaled_feature_statistics": scaled_stats,
    }


# ======================================================================
# BUILD FINAL DATASET
# ======================================================================

def build_risk_dataset(
    behaviour_df: pd.DataFrame,
    selected_features: List[str],
):
    print_header("BUILDING RISK DATASET")

    # --------------------------------------------------------------
    # Keep session ID.
    # --------------------------------------------------------------

    result = pd.DataFrame()

    result["session_id"] = (
        behaviour_df["session_id"]
        .astype(str)
    )

    # --------------------------------------------------------------
    # Preserve raw values.
    # --------------------------------------------------------------

    for feature in selected_features:

        result[feature] = (
            behaviour_df[feature]
            .astype(float)
        )

    # --------------------------------------------------------------
    # Transform.
    # --------------------------------------------------------------

    transformed_df, transformation_metadata = (
        transform_features(
            behaviour_df,
            selected_features,
        )
    )

    # --------------------------------------------------------------
    # Store transformed representation.
    #
    # These are NOT the final scaled values.
    # They are useful for reproducibility and analysis.
    # --------------------------------------------------------------

    for feature in selected_features:

        result[
            f"transformed_{feature}"
        ] = transformed_df[feature]

    # --------------------------------------------------------------
    # Scale.
    # --------------------------------------------------------------

    scaled_df, scaling_metadata = (
        robust_scale_features(
            transformed_df,
            selected_features,
        )
    )

    for column in scaled_df.columns:

        result[column] = scaled_df[column]

    return (
        result,
        transformation_metadata,
        scaling_metadata,
    )


# ======================================================================
# REPORT
# ======================================================================

def build_report(
    dataset: pd.DataFrame,
    selected_features: List[str],
    transformation_metadata: dict,
    scaling_metadata: dict,
    validation_report: dict,
) -> dict:

    raw_features = [
        f"raw_{feature}"
        for feature in selected_features
    ]

    transformed_features = [
        f"transformed_{feature}"
        for feature in selected_features
    ]

    scaled_features = [
        f"risk_scaled_{feature}"
        for feature in selected_features
    ]

    return {

        "project": "Adaptive Deception",

        "stage": "Risk Dataset Preparation",

        "purpose": (
            "Prepare reviewed behavioural features "
            "for future risk scoring."
        ),

        "rows": int(len(dataset)),

        "session_identifier": "session_id",

        "selected_feature_count": len(
            selected_features
        ),

        "selected_features": selected_features,

        "raw_feature_columns": raw_features,

        "transformed_feature_columns": transformed_features,

        "scaled_feature_columns": scaled_features,

        "transformations": transformation_metadata,

        "scaling": scaling_metadata,

        "validation": validation_report,

        "methodological_constraints": [

            "No attacker ground-truth labels were created.",

            "No attacker intent was inferred.",

            "No supervised attacker classifier was trained.",

            "The dataset represents behavioural risk-scoring candidates.",

            "Feature selection results should not be interpreted as proof of malicious intent.",

            "Risk scoring must be validated before deployment.",

        ],

    }


# ======================================================================
# MAIN
# ======================================================================

def main():

    print_header(
        "PREPARING BEHAVIOURAL RISK-SCORING DATASET"
    )

    ensure_directory()

    # ==============================================================
    # 1. LOAD SELECTED FEATURES
    # ==============================================================

    selected_df = load_csv(
        SELECTED_FEATURE_FILE,
        "selected risk features",
    )

    selected_features = extract_selected_features(
        selected_df
    )

    # ==============================================================
    # 2. LOAD BEHAVIOURAL DATASET
    # ==============================================================

    behaviour_df = load_csv(
        BEHAVIOUR_FILE,
        "behavioural dataset",
    )

    # ==============================================================
    # 3. VALIDATE
    # ==============================================================

    validate_input_features(
        behaviour_df,
        selected_features,
    )

    # ==============================================================
    # 4. CLEAN
    # ==============================================================

    behaviour_df = clean_numeric_values(
        behaviour_df,
        selected_features,
    )

    # ==============================================================
    # 5. BUILD DATASET
    # ==============================================================

    (
        risk_df,
        transformation_metadata,
        scaling_metadata,
    ) = build_risk_dataset(
        behaviour_df,
        selected_features,
    )

    # ==============================================================
    # 6. VALIDATE FINAL DATASET
    # ==============================================================

    scaled_features = [
        f"risk_scaled_{feature}"
        for feature in selected_features
    ]

    validation_report = validate_final_dataset(
        risk_df,
        selected_features,
        scaled_features,
    )

    # ==============================================================
    # 7. BUILD REPORT
    # ==============================================================

    report = build_report(
        risk_df,
        selected_features,
        transformation_metadata,
        scaling_metadata,
        validation_report,
    )

    # ==============================================================
    # 8. SAVE DATASET
    # ==============================================================

    print_header("SAVING RISK DATASET")

    risk_df.to_csv(
        OUTPUT_DATASET,
        index=False,
    )

    print("Saved:")
    print(OUTPUT_DATASET)

    # ==============================================================
    # 9. SAVE SCALING METADATA
    # ==============================================================

    save_json(
        OUTPUT_SCALING,
        {
            "selected_features": selected_features,
            "transformations": transformation_metadata,
            "scaling": scaling_metadata,
        },
    )

    print()
    print("Saved:")
    print(OUTPUT_SCALING)

    # ==============================================================
    # 10. SAVE REPORT
    # ==============================================================

    save_json(
        OUTPUT_REPORT,
        report,
    )

    print()
    print("Saved:")
    print(OUTPUT_REPORT)

    # ==============================================================
    # FINAL SUMMARY
    # ==============================================================

    print_header(
        "RISK DATASET PREPARATION COMPLETE"
    )

    print(
        f"Rows: {len(risk_df):,}"
    )

    print(
        f"Selected features: "
        f"{len(selected_features)}"
    )

    print(
        f"Output columns: "
        f"{len(risk_df.columns)}"
    )

    print()
    print("Selected features:")

    for index, feature in enumerate(
        selected_features,
        start=1,
    ):
        print(
            f"{index:2d}. {feature}"
        )

    print()
    print("Generated files:")

    print(
        f"1. {OUTPUT_DATASET}"
    )

    print(
        f"2. {OUTPUT_SCALING}"
    )

    print(
        f"3. {OUTPUT_REPORT}"
    )

    print()
    print("=" * 70)
    print("IMPORTANT")
    print("=" * 70)

    print(
        "This dataset is prepared for the future "
        "risk-scoring stage."
    )

    print(
        "It does NOT represent ground-truth attacker labels."
    )

    print(
        "Do NOT interpret the risk score as proof of attacker intent."
    )

    print(
        "Do NOT deploy adaptive deception policies yet."
    )


if __name__ == "__main__":
    main()