import gzip
import json
from pathlib import Path
from datetime import datetime

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")

OUTPUT_DIR = (
    DATA_DIR / "processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "sequence_features.csv"
)


# ============================================================
# COWRIE EVENT TYPES
# ============================================================

FAILED_LOGIN = "cowrie.login.failed"

SUCCESS_LOGIN = "cowrie.login.success"

COMMAND_EVENT = "cowrie.command.input"

FILE_EVENTS = {
    "cowrie.session.file_download",
    "cowrie.session.file_upload",
}


# ============================================================
# FIND RAW FILES
# ============================================================

def find_raw_files():

    files = sorted(
        DATA_DIR.glob("cyberlab_*.json.gz")
    )

    print()
    print("=" * 70)
    print("RAW COWRIE DATA")
    print("=" * 70)

    print(
        f"Files found: {len(files)}"
    )

    for file in files:
        print(
            f"  - {file.name}"
        )

    if not files:

        raise FileNotFoundError(
            "No cyberlab_*.json.gz files found."
        )

    return files


# ============================================================
# LOAD GZIP JSON
# ============================================================

def load_json_gzip(file_path):

    print()
    print(
        f"Loading {file_path.name}..."
    )

    with gzip.open(
        file_path,
        "rt",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    print(
        f"Top-level records: {len(data):,}"
    )

    return data


# ============================================================
# PARSE TIMESTAMP
# ============================================================

def parse_timestamp(value):

    if not value:

        return None

    try:

        return datetime.fromisoformat(
            value.replace(
                "Z",
                "+00:00"
            )
        )

    except (
        ValueError,
        TypeError
    ):

        return None


# ============================================================
# SORT EVENTS
# ============================================================

def sort_events(events):

    return sorted(
        events,
        key=lambda event:
        parse_timestamp(
            event.get("timestamp")
        )
        or datetime.min
    )


# ============================================================
# COUNT EVENT TYPE
# ============================================================

def count_event_type(
    events,
    event_type
):

    return sum(
        1
        for event in events
        if event.get("eventid")
        == event_type
    )


# ============================================================
# COUNT FILE EVENTS
# ============================================================

def count_file_events(events):

    return sum(
        1
        for event in events
        if event.get("eventid")
        in FILE_EVENTS
    )


# ============================================================
# BUILD EVENT SEQUENCE
# ============================================================

def build_event_sequence(events):

    return " -> ".join(
        event.get(
            "eventid",
            "unknown"
        )
        for event in events
    )


# ============================================================
# COUNT UNIQUE EVENT TYPES
# ============================================================

def count_unique_event_types(events):

    event_types = {
        event.get("eventid")
        for event in events
        if event.get("eventid")
    }

    return len(event_types)


# ============================================================
# COUNT UNIQUE TRANSITIONS
# ============================================================

def count_unique_transitions(events):

    transitions = set()

    for i in range(
        len(events) - 1
    ):

        current_event = events[i].get(
            "eventid"
        )

        next_event = events[i + 1].get(
            "eventid"
        )

        if current_event and next_event:

            transitions.add(
                (
                    current_event,
                    next_event
                )
            )

    return len(transitions)


# ============================================================
# TIME TO FIRST EVENT TYPE
# ============================================================

def time_to_event(
    events,
    target_event_types
):

    if not events:

        return None

    first_time = parse_timestamp(
        events[0].get(
            "timestamp"
        )
    )

    if first_time is None:

        return None

    for event in events:

        event_type = event.get(
            "eventid"
        )

        if event_type not in target_event_types:

            continue

        event_time = parse_timestamp(
            event.get(
                "timestamp"
            )
        )

        if event_time is None:

            continue

        return (
            event_time -
            first_time
        ).total_seconds()

    return None


# ============================================================
# FIND SUCCESSFUL LOGIN INDEX
# ============================================================

def find_successful_login_index(events):

    for index, event in enumerate(events):

        if event.get(
            "eventid"
        ) == SUCCESS_LOGIN:

            return index

    return None


# ============================================================
# COUNT EVENTS AFTER LOGIN
# ============================================================

def count_after_successful_login(
    events,
    target_event_types
):

    login_index = find_successful_login_index(
        events
    )

    if login_index is None:

        return 0

    return sum(
        1
        for event in events[
            login_index + 1:
        ]
        if event.get("eventid")
        in target_event_types
    )


# ============================================================
# DETERMINE SESSION STAGE
# ============================================================

def determine_session_stage(events):

    event_types = {
        event.get("eventid")
        for event in events
    }

    if event_types.intersection(
        FILE_EVENTS
    ):

        return "file_activity"

    if COMMAND_EVENT in event_types:

        return "command_activity"

    if SUCCESS_LOGIN in event_types:

        return "authenticated"

    if FAILED_LOGIN in event_types:

        return "authentication_attempt"

    return "connection"


# ============================================================
# EXTRACT FEATURES FROM ONE SESSION
# ============================================================

def extract_session_features(
    session_id,
    events
):

    if not events:

        return None

    events = sort_events(
        events
    )

    event_types = [
        event.get("eventid")
        for event in events
    ]

    event_types = [
        event
        for event in event_types
        if event
    ]

    if not event_types:

        return None

    # --------------------------------------------------------
    # Authentication
    # --------------------------------------------------------

    failed_logins = count_event_type(
        events,
        FAILED_LOGIN
    )

    successful_logins = count_event_type(
        events,
        SUCCESS_LOGIN
    )

    # --------------------------------------------------------
    # Commands
    # --------------------------------------------------------

    command_count = count_event_type(
        events,
        COMMAND_EVENT
    )

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    file_count = count_file_events(
        events
    )

    # --------------------------------------------------------
    # Feature record
    # --------------------------------------------------------

    record = {

        "session_id":
            session_id,

        "event_count":
            len(events),

        "first_event":
            event_types[0],

        "last_event":
            event_types[-1],

        "event_sequence":
            build_event_sequence(
                events
            ),

        "unique_event_types":
            count_unique_event_types(
                events
            ),

        "unique_event_transitions":
            count_unique_transitions(
                events
            ),

        "failed_login_count":
            failed_logins,

        "successful_login_count":
            successful_logins,

        "command_count":
            command_count,

        "file_event_count":
            file_count,

        "commands_after_login":
            count_after_successful_login(
                events,
                {COMMAND_EVENT}
            ),

        "files_after_login":
            count_after_successful_login(
                events,
                FILE_EVENTS
            ),

        "time_to_failed_login_sec":
            time_to_event(
                events,
                {FAILED_LOGIN}
            ),

        "time_to_successful_login_sec":
            time_to_event(
                events,
                {SUCCESS_LOGIN}
            ),

        "time_to_first_command_sec":
            time_to_event(
                events,
                {COMMAND_EVENT}
            ),

        "time_to_first_file_event_sec":
            time_to_event(
                events,
                FILE_EVENTS
            ),

        "session_stage":
            determine_session_stage(
                events
            )
    }

    return record


# ============================================================
# PROCESS ONE RAW FILE
# ============================================================

def process_file(file_path):

    data = load_json_gzip(
        file_path
    )

    records = []

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Your data is:
    #
    # [
    #     {
    #         "session_id": [
    #             event,
    #             event,
    #             ...
    #         ]
    #     },
    #     ...
    # ]
    #
    # --------------------------------------------------------

    for item in data:

        if not isinstance(
            item,
            dict
        ):

            continue

        for session_id, events in item.items():

            if not isinstance(
                events,
                list
            ):

                continue

            record = extract_session_features(
                session_id,
                events
            )

            if record is not None:

                records.append(
                    record
                )

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ADAPTIVE DECEPTION AI")
    print("SEQUENCE FEATURE EXTRACTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    raw_files = find_raw_files()

    all_records = []

    # --------------------------------------------------------
    # Process files
    # --------------------------------------------------------

    for file_index, file_path in enumerate(
        raw_files,
        start=1
    ):

        print()
        print(
            f"[{file_index}/{len(raw_files)}]"
        )

        records = process_file(
            file_path
        )

        print(
            f"Sessions extracted: "
            f"{len(records):,}"
        )

        all_records.extend(
            records
        )

        print(
            f"Total accumulated: "
            f"{len(all_records):,}"
        )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING DATASET")
    print("=" * 70)

    df = pd.DataFrame(
        all_records
    )

    print(
        f"Total records: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=[
            "session_id"
        ]
    )

    duplicates_removed = (
        before -
        len(df)
    )

    print(
        f"Duplicate sessions removed: "
        f"{duplicates_removed:,}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("SEQUENCE DATASET SAVED")
    print("=" * 70)

    print()
    print(
        f"Path:"
    )

    print(
        OUTPUT_FILE.resolve()
    )

    print()
    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Show sample
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SAMPLE RECORD")
    print("=" * 70)

    if not df.empty:

        sample = df.iloc[0]

        for column in df.columns:

            value = sample[column]

            if column == "event_sequence":

                value = str(value)[:500]

            print(
                f"{column}: {value}"
            )

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()