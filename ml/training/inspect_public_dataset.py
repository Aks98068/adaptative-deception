from pathlib import Path
import json
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "public"
    / "cic_ids2017"
    / "normalized_public_dataset.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "public"
    / "cic_ids2017"
)

SCHEMA_FILE = OUTPUT_DIR / "public_dataset_schema.json"
SUMMARY_FILE = OUTPUT_DIR / "public_dataset_summary.json"


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main():
    section("CIC-IDS2017 PUBLIC DATASET INSPECTION")

    print("Loading:")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Normalized public dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    section("COLUMN SCHEMA")

    for i, column in enumerate(df.columns, start=1):
        dtype = str(df[column].dtype)
        missing = int(df[column].isna().sum())
        unique = int(df[column].nunique(dropna=True))

        print(
            f"{i:3d}. "
            f"{column:<45} "
            f"dtype={dtype:<12} "
            f"unique={unique:<10,} "
            f"missing={missing:,}"
        )

    section("LABEL INFORMATION")

    if "label" in df.columns:
        print(df["label"].value_counts(dropna=False).to_string())

    if "is_attack" in df.columns:
        print()
        print("Binary attack distribution:")
        print(df["is_attack"].value_counts(dropna=False).to_string())

    section("NUMERIC FEATURES")

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    print(f"Numeric columns: {len(numeric_columns)}")

    for column in numeric_columns:
        series = pd.to_numeric(df[column], errors="coerce")

        print(
            f"{column:<45} "
            f"min={series.min():.6g} "
            f"max={series.max():.6g} "
            f"mean={series.mean():.6g}"
        )

    section("CATEGORICAL / TEXT FEATURES")

    categorical_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    print(f"Categorical/text columns: {len(categorical_columns)}")

    for column in categorical_columns:
        values = df[column].dropna().astype(str)

        print()
        print(f"{column}")
        print(f"Unique values: {values.nunique():,}")

        if values.nunique() <= 30:
            print(values.value_counts().head(30).to_string())

    section("LIKELY NETWORK-FLOW FEATURES")

    keywords = [
        "flow",
        "duration",
        "packet",
        "bytes",
        "length",
        "rate",
        "flag",
        "header",
        "forward",
        "backward",
        "bulk",
        "idle",
        "active",
        "subflow",
        "timestamp",
        "protocol",
        "port",
    ]

    network_candidates = []

    for column in df.columns:
        column_lower = column.lower()

        if any(keyword in column_lower for keyword in keywords):
            network_candidates.append(column)

    for column in network_candidates:
        print(f"- {column}")

    section("BEHAVIOURAL FEATURE MAPPING CANDIDATES")

    target_features = [
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
        "unique_event_transitions",
        "time_to_failed_login_sec",
        "time_to_successful_login_sec",
        "time_to_first_command_sec",
        "time_to_first_file_event_sec",
        "session_stage_encoded",
    ]

    print()

    for feature in target_features:
        print(f"{feature:<40} -> REVIEW REQUIRED")

    section("IMPORTANT DATASET LIMITATIONS")

    limitations = [
        "CIC-IDS2017 is primarily a network-flow dataset.",
        "SSH usernames are not directly represented as behavioural fields.",
        "SSH authentication success/failure is not directly represented.",
        "Shell commands are not directly represented.",
        "File operations are not directly represented.",
        "HASSH is not directly represented.",
        "Some existing behavioural features therefore cannot be mapped honestly.",
        "Unavailable features must not be fabricated from unrelated network fields.",
        "Derived features must have a documented mathematical definition.",
        "The CIC-IDS2017 label can be retained as external ground truth.",
    ]

    for item in limitations:
        print(f"- {item}")

    section("BUILDING SCHEMA REPORT")

    schema = {
        "dataset": "CIC-IDS2017",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "columns_detail": [],
        "label_columns": [],
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "network_candidate_columns": network_candidates,
        "behavioural_features": target_features,
        "limitations": limitations,
    }

    for column in df.columns:
        series = df[column]

        schema["columns_detail"].append(
            {
                "name": column,
                "dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "unique_values": int(series.nunique(dropna=True)),
            }
        )

    if "label" in df.columns:
        schema["label_columns"].append("label")

    if "is_attack" in df.columns:
        schema["label_columns"].append("is_attack")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SCHEMA_FILE, "w", encoding="utf-8") as f:
        json.dump(
            schema,
            f,
            indent=2,
            ensure_ascii=False,
        )

    summary = {
        "dataset": "CIC-IDS2017",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": len(numeric_columns),
        "categorical_columns": len(categorical_columns),
        "network_candidate_columns": len(network_candidates),
        "label_columns": schema["label_columns"],
    }

    if "label" in df.columns:
        summary["label_distribution"] = {
            str(k): int(v)
            for k, v in df["label"].value_counts(dropna=False).items()
        }

    if "is_attack" in df.columns:
        summary["binary_attack_distribution"] = {
            str(k): int(v)
            for k, v in df["is_attack"].value_counts(dropna=False).items()
        }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    section("INSPECTION COMPLETE")

    print(f"Schema report saved:")
    print(SCHEMA_FILE)

    print()
    print("Summary report saved:")
    print(SUMMARY_FILE)

    print()
    print("NEXT STEP:")
    print(
        "Review the actual CIC-IDS2017 columns before creating "
        "the public behavioural feature mapping."
    )


if __name__ == "__main__":
    main()