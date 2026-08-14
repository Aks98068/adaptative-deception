import gzip
import json
from pathlib import Path
from collections import defaultdict


# Dataset directory
DATA_DIR = Path("data")


def load_file(file_path):
    """
    Load one compressed JSON.GZ dataset file.
    """

    print(f"Loading: {file_path.name}")

    with gzip.open(
        file_path,
        "rt",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data


def reconstruct_sessions(data):
    """
    Convert the raw dataset structure into:

    session_id -> list of events
    """

    sessions = defaultdict(list)

    for record in data:

        # Make sure the record is a dictionary
        if not isinstance(record, dict):
            continue

        # Extract session ID and events
        for session_id, events in record.items():

            # Make sure events are stored as a list
            if not isinstance(events, list):
                continue

            # Add events to this session
            sessions[session_id].extend(events)

    return sessions


def main():

    # Find all JSON.GZ files
    files = sorted(
        DATA_DIR.glob("*.json.gz")
    )

    print(
        f"Files found: {len(files)}"
    )

    all_sessions = {}

    # Process every dataset file
    for file_path in files:

        data = load_file(file_path)

        sessions = reconstruct_sessions(data)

        print(
            f"Sessions found in file: {len(sessions)}"
        )

        # Add sessions to global collection
        for session_id, events in sessions.items():

            if session_id not in all_sessions:

                all_sessions[session_id] = []

            all_sessions[session_id].extend(events)

    print()
    print("==============================")
    print("SESSION RECONSTRUCTION")
    print("==============================")

    print(
        "Total sessions:",
        len(all_sessions)
    )

    # Show one example session
    if all_sessions:

        first_session_id = next(
            iter(all_sessions)
        )

        first_events = (
            all_sessions[first_session_id]
        )

        print()
        print("Example session ID:")
        print(first_session_id)

        print()
        print("Number of events:")
        print(len(first_events))

        print()
        print("Event sequence:")

        for event in first_events:

            event_id = event.get(
                "eventid",
                "UNKNOWN"
            )

            print(
                f"  → {event_id}"
            )


if __name__ == "__main__":
    main()