from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "public"
    / "cic_ids2017"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "public"
    / "cic_ids2017"
)

OUTPUT_DATASET = OUTPUT_DIR / "normalized_public_dataset.csv"
OUTPUT_METADATA = OUTPUT_DIR / "dataset_metadata.json"


# ============================================================
# CONFIGURATION
# ============================================================

LABEL_CANDIDATES = [
    "label",
    "Label",
    "LABEL",
]

ATTACK_TYPE_CANDIDATES = [
    "attack",
    "Attack",
    "attack_type",
    "Attack Type",
    "attack_category",
    "Attack Category",
]


# ============================================================
# LOGGING
# ============================================================

def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_column_name(column: str) -> str:
    """
    Convert inconsistent CSV column names into stable snake_case names.
    """

    column = str(column).strip()

    # Remove BOM
    column = column.replace("\ufeff", "")

    # Normalize whitespace
    column = re.sub(r"\s+", "_", column)

    # Replace punctuation
    column = re.sub(r"[^a-zA-Z0-9_]+", "_", column)

    # Remove repeated underscores
    column = re.sub(r"_+", "_", column)

    # Remove leading/trailing underscores
    column = column.strip("_")

    return column.lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize all dataframe column names.
    """

    original_columns = list(df.columns)

    normalized_columns = [
        normalize_column_name(column)
        for column in original_columns
    ]

    # Handle duplicate column names after normalization.
    seen: Dict[str, int] = {}
    final_columns: List[str] = []

    for column in normalized_columns:
        if column not in seen:
            seen[column] = 0
            final_columns.append(column)
        else:
            seen[column] += 1
            final_columns.append(
                f"{column}_{seen[column]}"
            )

    df.columns = final_columns

    return df


# ============================================================
# CSV DISCOVERY
# ============================================================

def discover_csv_files() -> List[Path]:

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw CIC-IDS2017 directory does not exist:\n{RAW_DIR}"
        )

    files = sorted(
        path
        for path in RAW_DIR.glob("*.csv")
        if path.is_file()
    )

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in:\n{RAW_DIR}"
        )

    return files


# ============================================================
# LABEL DETECTION
# ============================================================

def detect_label_column(columns: List[str]) -> str | None:

    normalized = {
        column.lower(): column
        for column in columns
    }

    for candidate in LABEL_CANDIDATES:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    # Fallback
    for column in columns:
        if column.lower() == "label":
            return column

    return None


def detect_attack_column(columns: List[str]) -> str | None:

    normalized = {
        column.lower(): column
        for column in columns
    }

    for candidate in ATTACK_TYPE_CANDIDATES:

        key = candidate.lower()

        if key in normalized:
            return normalized[key]

    return None


# ============================================================
# DATA CLEANING
# ============================================================

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # Remove completely empty rows
    df = df.dropna(axis=0, how="all")

    # Strip string values
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].astype(str).str.strip()

    return df


# ============================================================
# NUMERIC CLEANING
# ============================================================

def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:

    label_columns = {
        column
        for column in [
            detect_label_column(list(df.columns)),
            detect_attack_column(list(df.columns)),
        ]
        if column is not None
    }

    for column in df.columns:

        if column in label_columns:
            continue

        if df[column].dtype == "object":

            converted = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            numeric_ratio = converted.notna().mean()

            # Convert only when overwhelmingly numeric.
            if numeric_ratio >= 0.95:
                df[column] = converted

    return df


# ============================================================
# INF / NA HANDLING
# ============================================================

def clean_numeric_values(df: pd.DataFrame) -> pd.DataFrame:

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns

    if len(numeric_columns) == 0:
        return df

    df[numeric_columns] = df[numeric_columns].replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    return df


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_labels(
    df: pd.DataFrame,
    label_column: str,
) -> pd.DataFrame:

    df[label_column] = (
        df[label_column]
        .astype(str)
        .str.strip()
    )

    # Binary research label:
    #
    # BENIGN -> 0
    # everything else -> 1
    #
    # We preserve the original label separately.

    df["original_label"] = df[label_column]

    df["is_attack"] = (
        df[label_column]
        .str.upper()
        .ne("BENIGN")
        .astype("int8")
    )

    return df


# ============================================================
# FILE METADATA
# ============================================================

def inspect_file(
    path: Path,
) -> Dict:

    print()
    print(f"Reading:")
    print(path.name)

    try:

        df = pd.read_csv(
            path,
            low_memory=False,
        )

    except Exception as exc:

        raise RuntimeError(
            f"Unable to read CSV:\n{path}\n\n{exc}"
        ) from exc

    original_rows = len(df)
    original_columns = len(df.columns)

    df = normalize_columns(df)
    df = clean_dataframe(df)
    df = convert_numeric_columns(df)
    df = clean_numeric_values(df)

    label_column = detect_label_column(
        list(df.columns)
    )

    attack_column = detect_attack_column(
        list(df.columns)
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        f"Label column: {label_column}"
    )

    print(
        f"Attack column: {attack_column}"
    )

    return {
        "file": path.name,
        "original_rows": original_rows,
        "original_columns": original_columns,
        "processed_rows": len(df),
        "processed_columns": len(df.columns),
        "label_column": label_column,
        "attack_column": attack_column,
        "columns": list(df.columns),
    }


# ============================================================
# LOAD SINGLE DATASET
# ============================================================

def load_dataset_file(
    path: Path,
) -> pd.DataFrame:

    print()
    print("-" * 70)
    print(f"Loading: {path.name}")
    print("-" * 70)

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    df = normalize_columns(df)
    df = clean_dataframe(df)
    df = convert_numeric_columns(df)
    df = clean_numeric_values(df)

    label_column = detect_label_column(
        list(df.columns)
    )

    if label_column is None:
        raise ValueError(
            f"Could not detect the Label column in:\n{path}"
        )

    df = normalize_labels(
        df,
        label_column,
    )

    # Track source file.
    df["source_file"] = path.name

    return df


# ============================================================
# ALIGN DATAFRAMES
# ============================================================

def align_dataframes(
    dataframes: List[pd.DataFrame],
) -> List[pd.DataFrame]:

    all_columns = sorted(
        set().union(
            *(set(df.columns) for df in dataframes)
        )
    )

    aligned = []

    for df in dataframes:

        missing = [
            column
            for column in all_columns
            if column not in df.columns
        ]

        for column in missing:
            df[column] = pd.NA

        df = df[all_columns]

        aligned.append(df)

    return aligned


# ============================================================
# FINAL DATASET VALIDATION
# ============================================================

def validate_dataset(
    df: pd.DataFrame,
    label_column: str,
) -> Dict:

    section("FINAL DATASET VALIDATION")

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    missing_total = int(
        df.isna().sum().sum()
    )

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns

    infinite_total = 0

    if len(numeric_columns) > 0:

        infinite_total = int(
            df[numeric_columns]
            .isin([float("inf"), float("-inf")])
            .sum()
            .sum()
        )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    print(
        f"Missing values: {missing_total:,}"
    )

    print(
        f"Infinite values: {infinite_total:,}"
    )

    print(
        f"Duplicate rows: {duplicate_rows:,}"
    )

    print()
    print("LABEL DISTRIBUTION")
    print()

    print(
        df[label_column]
        .value_counts(dropna=False)
        .to_string()
    )

    print()
    print("BINARY ATTACK DISTRIBUTION")
    print()

    print(
        df["is_attack"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_values": missing_total,
        "infinite_values": infinite_total,
        "duplicate_rows": duplicate_rows,
        "label_distribution": {
            str(key): int(value)
            for key, value in df[label_column]
            .value_counts(dropna=False)
            .items()
        },
        "binary_attack_distribution": {
            str(key): int(value)
            for key, value in df["is_attack"]
            .value_counts()
            .items()
        },
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section("CIC-IDS2017 PUBLIC DATASET INGESTION")

    print(
        f"Input directory:\n{RAW_DIR}"
    )

    print(
        f"Output directory:\n{OUTPUT_DIR}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Discover files
    # --------------------------------------------------------

    section("DISCOVERING CSV FILES")

    files = discover_csv_files()

    print(
        f"CSV files discovered: {len(files)}"
    )

    for index, path in enumerate(files, start=1):

        print(
            f"{index:2d}. {path.name}"
        )

    # --------------------------------------------------------
    # Inspect files
    # --------------------------------------------------------

    section("INSPECTING DATASET FILES")

    file_metadata = []

    for path in files:

        metadata = inspect_file(path)

        file_metadata.append(metadata)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    section("LOADING DATASETS")

    dataframes = []

    for path in files:

        df = load_dataset_file(path)

        dataframes.append(df)

        print(
            f"Loaded rows: {len(df):,}"
        )

    # --------------------------------------------------------
    # Align columns
    # --------------------------------------------------------

    section("ALIGNING DATASET SCHEMAS")

    dataframes = align_dataframes(
        dataframes
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    section("COMBINING PUBLIC DATASET")

    combined = pd.concat(
        dataframes,
        ignore_index=True,
    )

    print(
        f"Combined rows: {len(combined):,}"
    )

    print(
        f"Combined columns: {len(combined.columns)}"
    )

    # --------------------------------------------------------
    # Determine label
    # --------------------------------------------------------

    label_column = detect_label_column(
        list(combined.columns)
    )

    if label_column is None:
        raise ValueError(
            "Final dataset does not contain a detectable label column."
        )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validation = validate_dataset(
        combined,
        label_column,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    section("SAVING NORMALIZED PUBLIC DATASET")

    combined.to_csv(
        OUTPUT_DATASET,
        index=False,
    )

    print(
        f"Saved:\n{OUTPUT_DATASET}"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "dataset": {
            "name": "CIC-IDS2017",
            "source_type": "public_cybersecurity_dataset",
            "input_directory": str(RAW_DIR),
            "output_dataset": str(
                OUTPUT_DATASET
            ),
        },
        "files": file_metadata,
        "combined": {
            "rows": int(len(combined)),
            "columns": int(len(combined.columns)),
            "column_names": list(combined.columns),
        },
        "labels": {
            "label_column": label_column,
            "binary_attack_column": "is_attack",
            "benign_definition": "BENIGN",
            "attack_definition": "all non-BENIGN labels",
        },
        "validation": validation,
        "methodological_notes": [
            "The original dataset labels are preserved.",
            "is_attack is a binary research label.",
            "No behavioural risk score is generated at this stage.",
            "No attacker intent is inferred from labels.",
            "Network-flow features are not assumed to be equivalent to session behavioural features.",
        ],
    }

    with open(
        OUTPUT_METADATA,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Saved:\n{OUTPUT_METADATA}"
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    section("PUBLIC DATASET INGESTION COMPLETE")

    print(
        f"Rows: {len(combined):,}"
    )

    print(
        f"Columns: {len(combined.columns)}"
    )

    print(
        f"Original label column: {label_column}"
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Inspect the normalized CIC-IDS2017 schema before designing"
    )

    print(
        "the network-to-behaviour feature mapping."
    )


if __name__ == "__main__":
    main()