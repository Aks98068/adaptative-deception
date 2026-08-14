from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/behavior_dataset.csv"
)

OUTPUT_FILE = Path(
    "data/processed/research_features.csv"
)


FEATURES = [

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

    "command_entropy",

    "num_file_events",

    "time_to_failed_login_sec",

    "time_to_successful_login_sec",

    "time_to_first_command_sec",

    "time_to_first_file_event_sec",

    "session_stage_encoded"
]


def main():

    print("=" * 70)
    print("CREATING RESEARCH FEATURE DATASET")
    print("=" * 70)

    print()
    print("Loading behavioural dataset...")

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Input shape: {df.shape}"
    )

    print()
    print("Selecting research features...")

    research_df = df[
        ["session_id"] + FEATURES
    ].copy()

    print(
        f"Output shape: {research_df.shape}"
    )

    print()
    print("Selected features:")

    for index, feature in enumerate(
        FEATURES,
        start=1
    ):

        print(
            f"{index:2}. {feature}"
        )

    print()
    print("Checking missing values...")

    missing = (
        research_df[FEATURES]
        .isnull()
        .sum()
        .sum()
    )

    print(
        f"Missing values: {missing}"
    )

    print()
    print("Checking infinite values...")

    numeric = research_df[
        FEATURES
    ]

    infinite = (
        numeric.isin(
            [float("inf"), float("-inf")]
        )
        .sum()
        .sum()
    )

    print(
        f"Infinite values: {infinite}"
    )

    print()
    print("Saving dataset...")

    research_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("RESEARCH DATASET CREATED")
    print("=" * 70)

    print()
    print(
        f"Rows: {len(research_df):,}"
    )

    print(
        f"Features: {len(FEATURES)}"
    )

    print()
    print(
        f"Saved to:\n{OUTPUT_FILE.resolve()}"
    )


if __name__ == "__main__":

    main()