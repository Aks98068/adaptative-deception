from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

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

OUTPUT_FILE = OUTPUT_DIR / "public_behaviour_dataset.csv"
REPORT_FILE = OUTPUT_DIR / "public_behaviour_dataset_report.json"


# ============================================================
# REQUIRED SOURCE COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "flow_duration",
    "total_fwd_packets",
    "total_backward_packets",
    "flow_iat_mean",
    "flow_iat_std",
    "flow_iat_min",
    "flow_iat_max",
    "flow_packets_s",
    "flow_bytes_s",
    "fwd_packets_s",
    "bwd_packets_s",
    "packet_length_mean",
    "packet_length_std",
    "packet_length_variance",
    "max_packet_length",
    "min_packet_length",
    "syn_flag_count",
    "fin_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "ack_flag_count",
    "urg_flag_count",
    "destination_port",
]


# ============================================================
# TARGET / LABEL COLUMNS
# ============================================================

LABEL_COLUMNS = [
    "label",
    "is_attack",
]


# ============================================================
# HELPERS
# ============================================================

def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def clean_column_names(df):
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    return df


def require_columns(df, columns):
    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Required CIC-IDS2017 columns are missing:\n"
            + "\n".join(f" - {x}" for x in missing)
        )


def numeric(df, column):
    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


def safe_divide(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    result = np.zeros_like(a, dtype=np.float64)

    mask = np.isfinite(a) & np.isfinite(b) & (b != 0)

    result[mask] = a[mask] / b[mask]

    return result


def clean_numeric_series(series):
    series = pd.to_numeric(
        series,
        errors="coerce",
    )

    series = series.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return series


# ============================================================
# MAIN FEATURE CONSTRUCTION
# ============================================================

def build_features(df):

    result = pd.DataFrame(index=df.index)

    # --------------------------------------------------------
    # BASIC FLOW COUNTS
    # --------------------------------------------------------

    fwd_packets = numeric(
        df,
        "total_fwd_packets",
    ).fillna(0)

    bwd_packets = numeric(
        df,
        "total_backward_packets",
    ).fillna(0)

    event_count = (
        fwd_packets + bwd_packets
    )

    result["event_count"] = event_count

    # --------------------------------------------------------
    # DURATION
    # CIC-IDS2017 Flow Duration is microseconds
    # --------------------------------------------------------

    flow_duration = numeric(
        df,
        "flow_duration",
    ).fillna(0)

    result["duration_sec"] = (
        flow_duration / 1_000_000.0
    )

    # --------------------------------------------------------
    # AVERAGE EVENT INTERVAL
    # --------------------------------------------------------

    result["average_event_interval"] = safe_divide(
        flow_duration / 1_000_000.0,
        event_count,
    )

    # --------------------------------------------------------
    # EVENT INTERVAL VARIANCE
    #
    # CIC provides Flow IAT Std.
    # Variance = standard deviation squared.
    # --------------------------------------------------------

    flow_iat_std = numeric(
        df,
        "flow_iat_std",
    ).fillna(0)

    result["event_interval_variance"] = (
        flow_iat_std / 1_000_000.0
    ) ** 2

    # --------------------------------------------------------
    # NETWORK ACTIVITY FEATURES
    # --------------------------------------------------------

    result["flow_packets_per_sec"] = numeric(
        df,
        "flow_packets_s",
    )

    result["flow_bytes_per_sec"] = numeric(
        df,
        "flow_bytes_s",
    )

    result["forward_packets_per_sec"] = numeric(
        df,
        "fwd_packets_s",
    )

    result["backward_packets_per_sec"] = numeric(
        df,
        "bwd_packets_s",
    )

    # --------------------------------------------------------
    # PACKET SIZE BEHAVIOUR
    # --------------------------------------------------------

    result["packet_length_mean"] = numeric(
        df,
        "packet_length_mean",
    )

    result["packet_length_std"] = numeric(
        df,
        "packet_length_std",
    )

    result["packet_length_variance"] = numeric(
        df,
        "packet_length_variance",
    )

    result["max_packet_length"] = numeric(
        df,
        "max_packet_length",
    )

    result["min_packet_length"] = numeric(
        df,
        "min_packet_length",
    )

    # --------------------------------------------------------
    # TCP FLAG BEHAVIOUR
    # --------------------------------------------------------

    result["syn_flag_count"] = numeric(
        df,
        "syn_flag_count",
    )

    result["fin_flag_count"] = numeric(
        df,
        "fin_flag_count",
    )

    result["rst_flag_count"] = numeric(
        df,
        "rst_flag_count",
    )

    result["psh_flag_count"] = numeric(
        df,
        "psh_flag_count",
    )

    result["ack_flag_count"] = numeric(
        df,
        "ack_flag_count",
    )

    result["urg_flag_count"] = numeric(
        df,
        "urg_flag_count",
    )

    # --------------------------------------------------------
    # DESTINATION PORT
    # --------------------------------------------------------

    result["destination_port"] = numeric(
        df,
        "destination_port",
    )

    # --------------------------------------------------------
    # NETWORK TRANSITION PROXY
    #
    # This is NOT equivalent to the original
    # unique_event_transitions feature.
    #
    # It represents TCP control-state diversity.
    # --------------------------------------------------------

    flags = pd.concat(
        [
            result["syn_flag_count"],
            result["fin_flag_count"],
            result["rst_flag_count"],
            result["psh_flag_count"],
            result["ack_flag_count"],
            result["urg_flag_count"],
        ],
        axis=1,
    ).fillna(0)

    result["network_flag_activity"] = (
        (flags > 0)
        .sum(axis=1)
        .astype(np.int16)
    )

    # --------------------------------------------------------
    # NETWORK ACTIVITY INTENSITY
    # --------------------------------------------------------

    result["packet_activity_intensity"] = np.log1p(
        np.maximum(
            result["event_count"],
            0,
        )
    )

    # --------------------------------------------------------
    # FORWARD/BACKWARD BALANCE
    # --------------------------------------------------------

    result["directional_packet_balance"] = safe_divide(
        fwd_packets - bwd_packets,
        fwd_packets + bwd_packets,
    )

    # --------------------------------------------------------
    # TCP SYN / RST BEHAVIOUR
    # --------------------------------------------------------

    result["connection_reset_ratio"] = safe_divide(
        result["rst_flag_count"],
        result["event_count"],
    )

    result["connection_attempt_ratio"] = safe_divide(
        result["syn_flag_count"],
        result["event_count"],
    )

    # --------------------------------------------------------
    # CLEAN ALL NUMERIC VALUES
    # --------------------------------------------------------

    for column in result.columns:
        result[column] = clean_numeric_series(
            result[column]
        )

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    result = result.fillna(0)

    return result


# ============================================================
# VALIDATION
# ============================================================

def validate_features(feature_df):

    section("VALIDATING PUBLIC BEHAVIOURAL FEATURES")

    if feature_df.empty:
        raise ValueError(
            "Generated public behavioural dataset is empty."
        )

    numeric_columns = feature_df.select_dtypes(
        include=["number"]
    ).columns

    missing = int(
        feature_df[numeric_columns]
        .isna()
        .sum()
        .sum()
    )

    infinite = int(
        np.isinf(
            feature_df[numeric_columns]
            .to_numpy()
        ).sum()
    )

    print(
        f"Rows: {len(feature_df):,}"
    )

    print(
        f"Features: {len(feature_df.columns)}"
    )

    print(
        f"Missing values: {missing:,}"
    )

    print(
        f"Infinite values: {infinite:,}"
    )

    if missing != 0:
        raise ValueError(
            "Missing values remain after feature construction."
        )

    if infinite != 0:
        raise ValueError(
            "Infinite values remain after feature construction."
        )

    print("Public behavioural feature validation complete.")


# ============================================================
# REPORT
# ============================================================

def build_report(
    source_df,
    feature_df,
):

    report = {
        "dataset": {
            "name": "CIC-IDS2017",
            "source_file": str(INPUT_FILE),
            "source_rows": int(len(source_df)),
            "generated_rows": int(len(feature_df)),
        },

        "feature_count": int(
            len(feature_df.columns)
        ),

        "features": list(
            feature_df.columns
        ),

        "ground_truth": {
            "label_column": (
                "label"
                if "label" in source_df.columns
                else None
            ),

            "binary_attack_column": (
                "is_attack"
                if "is_attack" in source_df.columns
                else None
            ),

            "label_distribution": {},
        },

        "feature_mapping": {
            "duration_sec":
                "flow_duration / 1,000,000",

            "event_count":
                "total_fwd_packets + total_backward_packets",

            "average_event_interval":
                "flow_duration_seconds / event_count",

            "event_interval_variance":
                "(flow_iat_std / 1,000,000)^2",

            "network_flag_activity":
                "number of active TCP flag types",

            "packet_activity_intensity":
                "log1p(event_count)",

            "directional_packet_balance":
                "(forward_packets - backward_packets) / total_packets",

            "connection_reset_ratio":
                "rst_flag_count / event_count",

            "connection_attempt_ratio":
                "syn_flag_count / event_count",
        },

        "unavailable_original_features": [
            "num_commands",
            "command_entropy",
            "failed_login_count",
            "successful_login_count",
            "failed_auth_ratio",
            "unique_usernames",
            "unique_hassh",
            "num_file_events",
            "time_to_failed_login_sec",
            "time_to_successful_login_sec",
            "time_to_first_command_sec",
            "time_to_first_file_event_sec",
        ],

        "research_constraints": [
            "CIC-IDS2017 is a network-flow dataset.",
            "Unavailable session-level authentication and command features were not fabricated.",
            "Derived features are explicitly documented.",
            "The original CIC-IDS2017 attack labels are retained as external ground truth.",
            "This dataset is intended for external validation.",
        ],
    }

    if "label" in source_df.columns:

        report["ground_truth"]["label_distribution"] = {
            str(k): int(v)
            for k, v in source_df[
                "label"
            ].value_counts(
                dropna=False
            ).items()
        }

    return report


# ============================================================
# MAIN
# ============================================================

def main():

    section("PUBLIC BEHAVIOURAL DATASET PREPARATION")

    print("Input:")
    print(INPUT_FILE)

    print()
    print("Output:")
    print(OUTPUT_FILE)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Normalized CIC-IDS2017 dataset not found:\n"
            f"{INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    section("LOADING NORMALIZED CIC-IDS2017")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # NORMALIZE COLUMN NAMES
    # --------------------------------------------------------

    df = clean_column_names(df)

    # --------------------------------------------------------
    # VALIDATE SOURCE
    # --------------------------------------------------------

    section("VALIDATING SOURCE COLUMNS")

    require_columns(
        df,
        REQUIRED_COLUMNS,
    )

    print(
        f"Required source columns present: "
        f"{len(REQUIRED_COLUMNS)}"
    )

    # --------------------------------------------------------
    # BUILD FEATURES
    # --------------------------------------------------------

    section("BUILDING PUBLIC NETWORK-BEHAVIOURAL FEATURES")

    feature_df = build_features(df)

    for column in feature_df.columns:
        print(
            f"Processed: {column}"
        )

    # --------------------------------------------------------
    # VALIDATE FEATURES
    # --------------------------------------------------------

    validate_features(
        feature_df
    )

    # --------------------------------------------------------
    # ATTACH GROUND TRUTH
    # --------------------------------------------------------

    section("ATTACHING EXTERNAL GROUND TRUTH")

    final_df = pd.DataFrame()

    # Preserve original row identity.
    final_df["public_row_id"] = np.arange(
        len(df),
        dtype=np.int64,
    )

    # Preserve CIC labels.
    if "label" in df.columns:
        final_df["label"] = (
            df["label"]
            .astype(str)
            .str.strip()
        )

    if "is_attack" in df.columns:
        final_df["is_attack"] = (
            pd.to_numeric(
                df["is_attack"],
                errors="coerce",
            )
            .fillna(0)
            .astype(np.int8)
        )

    # Add generated features.
    for column in feature_df.columns:
        final_df[column] = feature_df[column]

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    section("FINAL PUBLIC DATASET VALIDATION")

    numeric_columns = final_df.select_dtypes(
        include=["number"]
    ).columns

    missing_count = int(
        final_df[numeric_columns]
        .isna()
        .sum()
        .sum()
    )

    infinite_count = int(
        np.isinf(
            final_df[numeric_columns]
            .to_numpy()
        ).sum()
    )

    print(
        f"Rows: {len(final_df):,}"
    )

    print(
        f"Columns: {len(final_df.columns)}"
    )

    print(
        f"Missing numeric values: {missing_count:,}"
    )

    print(
        f"Infinite numeric values: {infinite_count:,}"
    )

    if missing_count != 0:
        raise ValueError(
            "Final dataset contains missing numeric values."
        )

    if infinite_count != 0:
        raise ValueError(
            "Final dataset contains infinite numeric values."
        )

    # --------------------------------------------------------
    # SAVE DATASET
    # --------------------------------------------------------

    section("SAVING PUBLIC BEHAVIOURAL DATASET")

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("Saved:")
    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    section("SAVING PUBLIC DATASET REPORT")

    report = build_report(
        df,
        feature_df,
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Saved:")
    print(REPORT_FILE)

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    section("PUBLIC BEHAVIOURAL DATASET PREPARATION COMPLETE")

    print(
        f"Rows: {len(final_df):,}"
    )

    print(
        f"Features generated: {len(feature_df.columns)}"
    )

    print()
    print("Ground-truth columns preserved:")

    if "label" in final_df.columns:
        print("- label")

    if "is_attack" in final_df.columns:
        print("- is_attack")

    print()
    print("Generated files:")
    print(f"1. {OUTPUT_FILE}")
    print(f"2. {REPORT_FILE}")

    print()
    print("IMPORTANT:")
    print(
        "This dataset is for external validation of "
        "behavioural scoring."
    )

    print(
        "Unavailable authentication, command, username, "
        "HASSH, and file-operation features were NOT fabricated."
    )

    print(
        "CIC-IDS2017 labels are retained as independent "
        "ground truth."
    )


if __name__ == "__main__":
    main()