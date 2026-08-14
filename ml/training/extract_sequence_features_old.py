import json
from pathlib import Path
from collections import Counter

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    Path("data")
    / "processed"
    / "reconstructed_sessions.json"
)

EXISTING_FEATURES = (
    Path("data")
    / "processed"
    / "ml_features.csv"
)

OUTPUT_FILE = (
    Path("data")
    / "processed"
    / "sequence_features.csv"
)


# ============================================================
# EVENT CATEGORIES
# ============================================================

AUTH_FAILED_EVENTS = {
    "cowrie.login.failed"
}

AUTH_SUCCESS_EVENTS = {
    "cowrie.login.success"
}

CONNECT_EVENTS = {
    "cowrie.session.connect"
}

CLOSED_EVENTS = {
    "cowrie.session.closed"
}

COMMAND_EVENTS = {
    "cowrie.command.input"
}

FILE_EVENTS = {
    "cowrie.session.file_download",
    "cowrie.session.file_upload"
}

CLIENT_EVENTS = {
    "cowrie.client.version"
}

KEX_EVENTS = {
    "cowrie.client.kex"
}


# ============================================================
# LOAD RECONSTRUCTED SESSIONS
# ============================================================

def load_sessions():

    print()
    print("=" * 70)
    print("LOADING RECONSTRUCTED SESSIONS")
    print("=" * 70)
    print()

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Could not find:\n"
            f"{INPUT_FILE.resolve()}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        sessions = json.load(file)

    print(
        f"Sessions loaded: {len(sessions):,}"
    )

    return sessions


# ============================================================
# NORMALIZE SESSION STRUCTURE
# ============================================================

def normalize_session(session):

    # Depending on how reconstruction stored the
    # session, it may already be a list of events.

    if isinstance(session, list):

        return session

    # Some reconstructed datasets store:
    #
    # {
    #     "session_id": [...]
    # }
    #
    # Handle that structure too.

    if isinstance(session, dict):

        if "events" in session:

            return session["events"]

        # If dictionary contains one session key
        values = list(session.values())

        if len(values) == 1:

            if isinstance(values[0], list):

                return values[0]

    return []


# ============================================================
# SORT EVENTS
# ============================================================

def sort_events(events):

    return sorted(
        events,
        key=lambda event: event.get(
            "timestamp",
            ""
        )
    )


# ============================================================
# EXTRACT EVENT TYPES
# ============================================================

def get_event_types(events):

    return [
        event.get(
            "eventid",
            "unknown"
        )
        for event in events
    ]


# ============================================================
# CREATE EVENT SEQUENCE
# ============================================================

def create_sequence(event_types):

    return " -> ".join(
        event_types
    )


# ============================================================
# COUNT EVENT TRANSITIONS
# ============================================================

def count_transitions(event_types):

    if len(event_types) < 2:

        return 0

    transitions = set()

    for i in range(
        len(event_types) - 1
    ):

        transition = (
            event_types[i],
            event_types[i + 1]
        )

        transitions.add(
            transition
        )

    return len(transitions)


# ============================================================
# TIME BETWEEN EVENTS
# ============================================================

def calculate_time_to_event(
    events,
    target_events
):

    if not events:

        return None

    first_timestamp = pd.to_datetime(
        events[0].get(
            "timestamp"
        ),
        errors="coerce"
    )

    if pd.isna(first_timestamp):

        return None

    for event in events:

        event_id = event.get(
            "eventid"
        )

        if event_id in target_events:

            timestamp = pd.to_datetime(
                event.get(
                    "timestamp"
                ),
                errors="coerce"
            )

            if pd.isna(timestamp):

                continue

            delta = (
                timestamp -
                first_timestamp
            )

            return delta.total_seconds()

    return None


# ============================================================
# FIND FIRST EVENT
# ============================================================

def first_matching_event(
    event_types,
    target_events
):

    for event_type in event_types:

        if event_type in target_events:

            return event_type

    return None


# ============================================================
# COUNT EVENTS AFTER AN EVENT
# ============================================================

def count_after_event(
    event_types,
    trigger_events,
    target_events
):

    trigger_index = None

    for index, event_type in enumerate(
        event_types
    ):

        if event_type in trigger_events:

            trigger_index = index

            break

    if trigger_index is None:

        return 0

    return sum(
        1
        for event_type
        in event_types[
            trigger_index + 1:
        ]
        if event_type in target_events
    )


# ============================================================
# DETERMINE SESSION STAGE
# ============================================================

def determine_session_stage(
    event_types
):

    has_failed_login = any(
        event in AUTH_FAILED_EVENTS
        for event in event_types
    )

    has_successful_login = any(
        event in AUTH_SUCCESS_EVENTS
        for event in event_types
    )

    has_commands = any(
        event in COMMAND_EVENTS
        for event in event_types
    )

    has_files = any(
        event in FILE_EVENTS
        for event in event_types
    )

    if has_files:

        return "file_activity"

    if has_commands:

        return "command_activity"

    if has_successful_login:

        return "authenticated"

    if has_failed_login:

        return "authentication_attempt"

    return "connection"


# ============================================================
# EXTRACT ONE SESSION
# ============================================================

def extract_features(
    session_id,
    events
):

    events = sort_events(
        events
    )

    event_types = get_event_types(
        events
    )

    if not event_types:

        return None

    # --------------------------------------------------------
    # Basic sequence information
    # --------------------------------------------------------

    sequence = create_sequence(
        event_types
    )

    unique_events = len(
        set(event_types)
    )

    transition_count = count_transitions(
        event_types
    )

    # --------------------------------------------------------
    # Time-based features
    # --------------------------------------------------------

    time_to_failed_login = calculate_time_to_event(
        events,
        AUTH_FAILED_EVENTS
    )

    time_to_successful_login = calculate_time_to_event(
        events,
        AUTH_SUCCESS_EVENTS
    )

    time_to_first_command = calculate_time_to_event(
        events,
        COMMAND_EVENTS
    )

    time_to_first_file_event = calculate_time_to_event(
        events,
        FILE_EVENTS
    )

    # --------------------------------------------------------
    # Authentication
    # --------------------------------------------------------

    failed_logins = sum(
        1
        for event in event_types
        if event in AUTH_FAILED_EVENTS
    )

    successful_logins = sum(
        1
        for event in event_types
        if event in AUTH_SUCCESS_EVENTS
    )

    # --------------------------------------------------------
    # Post-authentication behaviour
    # --------------------------------------------------------

    commands_after_login = count_after_event(
        event_types,
        AUTH_SUCCESS_EVENTS,
        COMMAND_EVENTS
    )

    files_after_login = count_after_event(
        event_types,
        AUTH_SUCCESS_EVENTS,
        FILE_EVENTS
    )

    # --------------------------------------------------------
    # Stage
    # --------------------------------------------------------

    session_stage = determine_session_stage(
        event_types
    )

    # --------------------------------------------------------
    # Return feature record
    # --------------------------------------------------------

    return {

        "session_id":
            session_id,

        "first_event":
            event_types[0],

        "last_event":
            event_types[-1],

        "event_sequence":
            sequence,

        "unique_sequence_events":
            unique_events,

        "event_transition_count":
            transition_count,

        "failed_login_events":
            failed_logins,

        "successful_login_events":
            successful_logins,

        "time_to_failed_login_sec":
            time_to_failed_login,

        "time_to_successful_login_sec":
            time_to_successful_login,

        "time_to_first_command_sec":
            time_to_first_command,

        "time_to_first_file_event_sec":
            time_to_first_file_event,

        "commands_after_login":
            commands_after_login,

        "files_after_login":
            files_after_login,

        "session_stage":
            session_stage
    }


# ============================================================
# PROCESS ALL SESSIONS
# ============================================================

def process_sessions(sessions):

    print()
    print("=" * 70)
    print("EXTRACTING SEQUENCE FEATURES")
    print("=" * 70)
    print()

    records = []

    total = len(
        sessions
    )

    for index, session in enumerate(
        sessions
    ):

        # ----------------------------------------------------
        # Determine session ID
        # ----------------------------------------------------

        session_id = None
        events = None

        if isinstance(
            session,
            dict
        ):

            # Standard reconstructed format
            if "session_id" in session:

                session_id = session[
                    "session_id"
                ]

            if "events" in session:

                events = session[
                    "events"
                ]

            # Handle {session_id: events}
            if events is None:

                keys = list(
                    session.keys()
                )

                if len(keys) == 1:

                    session_id = keys[0]

                    events = session[
                        keys[0]
                    ]

        # ----------------------------------------------------
        # Handle list structure
        # ----------------------------------------------------

        elif isinstance(
            session,
            list
        ):

            events = session

            if events:

                session_id = events[0].get(
                    "session_id"
                )

        if not session_id:

            continue

        if not isinstance(
            events,
            list
        ):

            continue

        # ----------------------------------------------------
        # Extract
        # ----------------------------------------------------

        record = extract_features(
            session_id,
            events
        )

        if record is not None:

            records.append(
                record
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (index + 1) % 100_000
            == 0
        ):

            print(
                f"Processed "
                f"{index + 1:,} / "
                f"{total:,}"
            )

    return records


# ============================================================
# SAVE FEATURES
# ============================================================

def save_features(records):

    print()
    print("=" * 70)
    print("CREATING SEQUENCE FEATURE DATAFRAME")
    print("=" * 70)
    print()

    df = pd.DataFrame(
        records
    )

    print(
        f"Sequence feature records: "
        f"{len(df):,}"
    )

    print()

    print(
        "Columns:"
    )

    for column in df.columns:

        print(
            f"  - {column}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(
        f"Saved to:"
    )

    print(
        OUTPUT_FILE.resolve()
    )

    return df


# ============================================================
# DISPLAY SAMPLE
# ============================================================

def show_sample(df):

    print()
    print("=" * 70)
    print("SAMPLE SEQUENCE FEATURES")
    print("=" * 70)
    print()

    if df.empty:

        print(
            "No sequence records generated."
        )

        return

    sample = df.iloc[0]

    for column in df.columns:

        print(
            f"{column}: "
            f"{sample[column]}"
        )


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
    # 1. Load sessions
    # --------------------------------------------------------

    sessions = load_sessions()

    # --------------------------------------------------------
    # 2. Process
    # --------------------------------------------------------

    records = process_sessions(
        sessions
    )

    # --------------------------------------------------------
    # 3. Save
    # --------------------------------------------------------

    df = save_features(
        records
    )

    # --------------------------------------------------------
    # 4. Show sample
    # --------------------------------------------------------

    show_sample(
        df
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SEQUENCE FEATURE EXTRACTION COMPLETE")
    print("=" * 70)
    print()

    print(
        "NEXT STEP:"
    )

    print(
        "Merge sequence features with the "
        "existing 14 ML features."
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()