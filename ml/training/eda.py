import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    Path("data")
    / "processed"
    / "ml_features.csv"
)

OUTPUT_DIR = (
    Path("data")
    / "processed"
    / "eda"
)

# Number of rows used for visualisation.
#
# We have more than 1 million sessions.
# We don't need to plot every single row.
SAMPLE_SIZE = 100_000


# Create output directory
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print()
    print("=" * 70)
    print("LOADING ML DATASET")
    print("=" * 70)
    print()

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{INPUT_FILE.resolve()}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    return df


# ============================================================
# BASIC STATISTICS
# ============================================================

def basic_statistics(df):

    print()
    print("=" * 70)
    print("1. BASIC STATISTICS")
    print("=" * 70)
    print()

    numerical_columns = (
        df.select_dtypes(
            include=np.number
        ).columns
    )

    statistics = (
        df[numerical_columns]
        .describe()
        .T
    )

    print(
        statistics
    )

    statistics.to_csv(
        OUTPUT_DIR /
        "feature_statistics.csv"
    )


# ============================================================
# MEDIAN AND PERCENTILES
# ============================================================

def percentile_analysis(df):

    print()
    print("=" * 70)
    print("2. PERCENTILE ANALYSIS")
    print("=" * 70)
    print()

    numerical_columns = (
        df.select_dtypes(
            include=np.number
        ).columns
    )

    percentiles = (
        df[numerical_columns]
        .quantile(
            [
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99
            ]
        )
        .T
    )

    print(
        percentiles
    )

    percentiles.to_csv(
        OUTPUT_DIR /
        "feature_percentiles.csv"
    )


# ============================================================
# ZERO-RATE ANALYSIS
# ============================================================

def zero_rate_analysis(df):

    print()
    print("=" * 70)
    print("3. ZERO-RATE ANALYSIS")
    print("=" * 70)
    print()

    numerical_columns = (
        df.select_dtypes(
            include=np.number
        ).columns
    )

    results = []

    for column in numerical_columns:

        zero_count = (
            df[column] == 0
        ).sum()

        zero_percentage = (
            zero_count /
            len(df)
        ) * 100

        results.append({

            "feature": column,

            "zero_count":
                zero_count,

            "zero_percentage":
                zero_percentage

        })

        print(
            f"{column:35} "
            f"{zero_percentage:8.2f}%"
        )

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        OUTPUT_DIR /
        "zero_rate.csv",
        index=False
    )


# ============================================================
# DISTRIBUTION PLOTS
# ============================================================

def distribution_plots(df):

    print()
    print("=" * 70)
    print("4. CREATING DISTRIBUTION PLOTS")
    print("=" * 70)
    print()

    # Random sample for plotting
    if len(df) > SAMPLE_SIZE:

        sample = df.sample(
            n=SAMPLE_SIZE,
            random_state=42
        )

    else:

        sample = df

    plot_features = [

        "duration_sec",

        "average_event_interval",

        "event_interval_variance",

        "event_count",

        "num_commands",

        "command_entropy",

        "failed_login_count",

        "successful_login_count",

        "failed_auth_ratio",

        "unique_usernames",

        "unique_hassh",

        "num_file_events",

        "session_closed_events"

    ]

    for feature in plot_features:

        if feature not in sample.columns:

            continue

        plt.figure(
            figsize=(10, 6)
        )

        sns.histplot(
            sample[feature],
            bins=50,
            kde=False
        )

        plt.title(
            f"Distribution of {feature}"
        )

        plt.xlabel(
            feature
        )

        plt.ylabel(
            "Number of sessions"
        )

        plt.tight_layout()

        output_path = (
            OUTPUT_DIR /
            f"{feature}_distribution.png"
        )

        plt.savefig(
            output_path,
            dpi=150
        )

        plt.close()

        print(
            f"Saved: {output_path.name}"
        )


# ============================================================
# LOG-SCALE DISTRIBUTIONS
# ============================================================

def log_distribution_plots(df):

    print()
    print("=" * 70)
    print("5. CREATING LOG-SCALE DISTRIBUTIONS")
    print("=" * 70)
    print()

    if len(df) > SAMPLE_SIZE:

        sample = df.sample(
            n=SAMPLE_SIZE,
            random_state=42
        )

    else:

        sample = df

    features = [

        "duration_sec",

        "event_interval_variance",

        "event_count",

        "num_commands",

        "failed_login_count",

        "num_file_events"

    ]

    for feature in features:

        if feature not in sample.columns:

            continue

        # Remove negative values.
        values = sample[
            feature
        ].clip(
            lower=0
        )

        # log1p handles zero safely.
        values = np.log1p(
            values
        )

        plt.figure(
            figsize=(10, 6)
        )

        sns.histplot(
            values,
            bins=50
        )

        plt.title(
            f"log(1 + {feature})"
        )

        plt.xlabel(
            f"log1p({feature})"
        )

        plt.ylabel(
            "Number of sessions"
        )

        plt.tight_layout()

        output_path = (
            OUTPUT_DIR /
            f"{feature}_log_distribution.png"
        )

        plt.savefig(
            output_path,
            dpi=150
        )

        plt.close()

        print(
            f"Saved: {output_path.name}"
        )


# ============================================================
# CORRELATION MATRIX
# ============================================================

def correlation_analysis(df):

    print()
    print("=" * 70)
    print("6. CORRELATION ANALYSIS")
    print("=" * 70)
    print()

    numerical_columns = (
        df.select_dtypes(
            include=np.number
        ).columns
    )

    correlation = (
        df[numerical_columns]
        .corr()
    )

    print(
        correlation.round(3)
    )

    correlation.to_csv(
        OUTPUT_DIR /
        "correlation_matrix.csv"
    )

    # Plot
    plt.figure(
        figsize=(14, 12)
    )

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        square=True
    )

    plt.title(
        "Feature Correlation Matrix"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        "correlation_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(
        f"Saved: {output_path.name}"
    )


# ============================================================
# OUTLIER ANALYSIS
# ============================================================

def outlier_analysis(df):

    print()
    print("=" * 70)
    print("7. OUTLIER ANALYSIS")
    print("=" * 70)
    print()

    numerical_columns = (
        df.select_dtypes(
            include=np.number
        ).columns
    )

    results = []

    for column in numerical_columns:

        q1 = df[column].quantile(
            0.25
        )

        q3 = df[column].quantile(
            0.75
        )

        iqr = q3 - q1

        lower_bound = (
            q1 - 1.5 * iqr
        )

        upper_bound = (
            q3 + 1.5 * iqr
        )

        outliers = (
            (df[column] < lower_bound)
            |
            (df[column] > upper_bound)
        )

        count = outliers.sum()

        percentage = (
            count /
            len(df)
        ) * 100

        results.append({

            "feature":
                column,

            "q1":
                q1,

            "q3":
                q3,

            "iqr":
                iqr,

            "lower_bound":
                lower_bound,

            "upper_bound":
                upper_bound,

            "outlier_count":
                count,

            "outlier_percentage":
                percentage

        })

        print(
            f"{column:35} "
            f"{percentage:8.2f}% outliers"
        )

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        OUTPUT_DIR /
        "outlier_analysis.csv",
        index=False
    )


# ============================================================
# BEHAVIOURAL SUMMARY
# ============================================================

def behavioural_summary(df):

    print()
    print("=" * 70)
    print("8. BEHAVIOURAL SUMMARY")
    print("=" * 70)
    print()

    total = len(df)

    # Sessions with commands
    command_sessions = (
        df["num_commands"] > 0
    ).sum()

    # Sessions with file activity
    file_sessions = (
        df["num_file_events"] > 0
    ).sum()

    # Sessions with successful login
    successful_sessions = (
        df["successful_login_count"] > 0
    ).sum()

    # Sessions with failed authentication
    failed_sessions = (
        df["failed_login_count"] > 0
    ).sum()

    # Longer sessions
    long_sessions = (
        df["duration_sec"] > 60
    ).sum()

    print(
        f"Total sessions: "
        f"{total:,}"
    )

    print()

    print(
        f"Sessions with commands: "
        f"{command_sessions:,} "
        f"({command_sessions / total * 100:.2f}%)"
    )

    print(
        f"Sessions with file activity: "
        f"{file_sessions:,} "
        f"({file_sessions / total * 100:.2f}%)"
    )

    print(
        f"Sessions with successful login: "
        f"{successful_sessions:,} "
        f"({successful_sessions / total * 100:.2f}%)"
    )

    print(
        f"Sessions with failed login: "
        f"{failed_sessions:,} "
        f"({failed_sessions / total * 100:.2f}%)"
    )

    print(
        f"Sessions longer than 60 seconds: "
        f"{long_sessions:,} "
        f"({long_sessions / total * 100:.2f}%)"
    )


# ============================================================
# HIGH-CORRELATION PAIRS
# ============================================================

def high_correlation_pairs(df):

    print()
    print("=" * 70)
    print("9. HIGH-CORRELATION FEATURE PAIRS")
    print("=" * 70)
    print()

    numerical_columns = (
        df.select_dtypes(
            include=np.number
        ).columns
    )

    correlation = (
        df[numerical_columns]
        .corr()
    )

    found = False

    for i in range(
        len(correlation.columns)
    ):

        for j in range(
            i + 1,
            len(correlation.columns)
        ):

            value = correlation.iloc[
                i,
                j
            ]

            if abs(value) >= 0.90:

                found = True

                print(
                    f"{correlation.columns[i]:35}"
                    f" <-> "
                    f"{correlation.columns[j]:35}"
                    f" = {value:.3f}"
                )

    if not found:

        print(
            "No correlation >= 0.90 found."
        )


# ============================================================
# SAVE SAMPLE DATA
# ============================================================

def save_sample(df):

    if len(df) > SAMPLE_SIZE:

        sample = df.sample(
            n=SAMPLE_SIZE,
            random_state=42
        )

    else:

        sample = df.copy()

    output_path = (
        OUTPUT_DIR /
        "eda_sample.csv"
    )

    sample.to_csv(
        output_path,
        index=False
    )

    print()
    print(
        f"EDA sample saved: "
        f"{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ADAPTIVE DECEPTION AI")
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    basic_statistics(
        df
    )

    # --------------------------------------------------------
    # Percentiles
    # --------------------------------------------------------

    percentile_analysis(
        df
    )

    # --------------------------------------------------------
    # Zero rate
    # --------------------------------------------------------

    zero_rate_analysis(
        df
    )

    # --------------------------------------------------------
    # Distribution plots
    # --------------------------------------------------------

    distribution_plots(
        df
    )

    # --------------------------------------------------------
    # Log distributions
    # --------------------------------------------------------

    log_distribution_plots(
        df
    )

    # --------------------------------------------------------
    # Correlation
    # --------------------------------------------------------

    correlation_analysis(
        df
    )

    # --------------------------------------------------------
    # Outliers
    # --------------------------------------------------------

    outlier_analysis(
        df
    )

    # --------------------------------------------------------
    # Behavioural summary
    # --------------------------------------------------------

    behavioural_summary(
        df
    )

    # --------------------------------------------------------
    # High correlations
    # --------------------------------------------------------

    high_correlation_pairs(
        df
    )

    # --------------------------------------------------------
    # Save sample
    # --------------------------------------------------------

    save_sample(
        df
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EDA COMPLETE")
    print("=" * 70)

    print()
    print(
        "EDA output directory:"
    )

    print(
        OUTPUT_DIR.resolve()
    )

    print()
    print(
        "Do not train the model yet."
    )

    print(
        "Review the EDA results first."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()