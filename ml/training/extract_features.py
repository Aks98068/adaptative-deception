import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

from reconstruct_sessions import (
    load_file,
    reconstruct_sessions,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Raw dataset directory
DATA_DIR = Path("data")

# Directory where processed datasets will be stored
OUTPUT_DIR = DATA_DIR / "processed"

# Create the output directory if it does not exist
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TIMESTAMP PARSER
# ============================================================

def parse_timestamp(timestamp):
    """
    Convert a Cowrie timestamp into a Python datetime object.

    Example:
        2020-02-29T00:06:51.433806Z
    """

    # If timestamp is missing
    if not timestamp:
        return None

    try:
        # Convert Z (UTC) into +00:00
        timestamp = timestamp.replace(
            "Z",
            "+00:00"
        )

        # Convert string into datetime
        return datetime.fromisoformat(
            timestamp
        )

    except (ValueError, TypeError):
        # Invalid timestamp
        return None


# ============================================================
# ENTROPY CALCULATION
# ============================================================

def calculate_entropy(values):
    """
    Calculate Shannon entropy.

    Higher entropy:
        More diverse behaviour.

    Lower entropy:
        More repetitive behaviour.
    """

    # No values
    if not values:
        return 0.0

    # Count occurrences
    counts = Counter(values)

    # Total number of values
    total = len(values)

    entropy = 0.0

    # Calculate Shannon entropy
    for count in counts.values():

        probability = count / total

        entropy -= (
            probability *
            math.log2(probability)
        )

    return entropy


# ============================================================
# EXTRACT FEATURES FROM ONE SESSION
# ============================================================

def extract_session_features(
    session_id,
    events
):
    """
    Convert one Cowrie session into
    numerical behavioural features.
    """

    # --------------------------------------------------------
    # Containers
    # --------------------------------------------------------

    timestamps = []

    event_ids = []

    usernames = []

    passwords = []

    hassh_values = []

    protocols = []

    commands = []

    file_events = []

    # --------------------------------------------------------
    # Authentication counters
    # --------------------------------------------------------

    failed_logins = 0

    successful_logins = 0

    # --------------------------------------------------------
    # Process every event
    # --------------------------------------------------------

    for event in events:

        # Make sure the event is a dictionary
        if not isinstance(event, dict):
            continue

        # ----------------------------------------------------
        # Event ID
        # ----------------------------------------------------

        event_id = event.get(
            "eventid"
        )

        if event_id:

            event_ids.append(
                event_id
            )

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        timestamp = parse_timestamp(
            event.get("timestamp")
        )

        if timestamp:

            timestamps.append(
                timestamp
            )

        # ----------------------------------------------------
        # Username
        # ----------------------------------------------------

        username = event.get(
            "username"
        )

        if username:

            usernames.append(
                username
            )

        # ----------------------------------------------------
        # Password
        # ----------------------------------------------------

        password = event.get(
            "password"
        )

        if password:

            passwords.append(
                password
            )

        # ----------------------------------------------------
        # SSH HASSH fingerprint
        # ----------------------------------------------------

        hassh = event.get(
            "hassh"
        )

        if hassh:

            hassh_values.append(
                hassh
            )

        # ----------------------------------------------------
        # Protocol
        # ----------------------------------------------------

        protocol = event.get(
            "protocol"
        )

        if protocol:

            protocols.append(
                protocol
            )

        # ----------------------------------------------------
        # Failed authentication
        # ----------------------------------------------------

        if event_id == "cowrie.login.failed":

            failed_logins += 1

        # ----------------------------------------------------
        # Successful authentication
        # ----------------------------------------------------

        elif event_id == "cowrie.login.success":

            successful_logins += 1

        # ----------------------------------------------------
        # Commands
        # ----------------------------------------------------

        if event_id == "cowrie.command.input":

            message = event.get(
                "message"
            )

            if message:

                commands.append(
                    message
                )

        # ----------------------------------------------------
        # File activity
        # ----------------------------------------------------

        if event_id in {
            "cowrie.session.file_download",
            "cowrie.session.file_upload",
        }:

            file_events.append(
                event
            )

    # ========================================================
    # TEMPORAL FEATURES
    # ========================================================

    # Sort timestamps chronologically
    timestamps.sort()

    # --------------------------------------------------------
    # Session duration
    # --------------------------------------------------------

    if len(timestamps) >= 2:

        start_time = timestamps[0]

        end_time = timestamps[-1]

        duration = (
            end_time - start_time
        ).total_seconds()

    else:

        duration = 0.0

    # --------------------------------------------------------
    # Time intervals between events
    # --------------------------------------------------------

    intervals = []

    if len(timestamps) >= 2:

        for index in range(
            1,
            len(timestamps)
        ):

            previous_time = (
                timestamps[index - 1]
            )

            current_time = (
                timestamps[index]
            )

            interval = (
                current_time -
                previous_time
            ).total_seconds()

            # Protect against negative values
            if interval >= 0:

                intervals.append(
                    interval
                )

    # --------------------------------------------------------
    # Average event interval
    # --------------------------------------------------------

    if intervals:

        average_interval = (
            sum(intervals) /
            len(intervals)
        )

    else:

        average_interval = 0.0

    # --------------------------------------------------------
    # Event interval variance
    # --------------------------------------------------------

    if intervals:

        variance = (
            sum(
                (
                    interval -
                    average_interval
                ) ** 2
                for interval in intervals
            )
            /
            len(intervals)
        )

    else:

        variance = 0.0

    # ========================================================
    # AUTHENTICATION FEATURES
    # ========================================================

    total_auth_attempts = (
        failed_logins +
        successful_logins
    )

    # --------------------------------------------------------
    # Failed authentication ratio
    # --------------------------------------------------------

    if total_auth_attempts > 0:

        failed_auth_ratio = (
            failed_logins /
            total_auth_attempts
        )

    else:

        failed_auth_ratio = 0.0

    # ========================================================
    # EVENT COUNTS
    # ========================================================

    event_counts = Counter(
        event_ids
    )

    # ========================================================
    # COMMAND FEATURES
    # ========================================================

    command_entropy = calculate_entropy(
        commands
    )

    # ========================================================
    # CREATE FINAL FEATURE RECORD
    # ========================================================

    features = {

        # ----------------------------------------------------
        # Session identification
        # ----------------------------------------------------

        "session_id":
            session_id,

        # ----------------------------------------------------
        # Temporal behaviour
        # ----------------------------------------------------

        "duration_sec":
            duration,

        "average_event_interval":
            average_interval,

        "event_interval_variance":
            variance,

        # ----------------------------------------------------
        # General session behaviour
        # ----------------------------------------------------

        "event_count":
            len(events),

        "unique_event_types":
            len(set(event_ids)),

        # ----------------------------------------------------
        # Command behaviour
        # ----------------------------------------------------

        "num_commands":
            len(commands),

        "command_entropy":
            command_entropy,

        # ----------------------------------------------------
        # Authentication behaviour
        # ----------------------------------------------------

        "failed_login_count":
            failed_logins,

        "successful_login_count":
            successful_logins,

        "total_auth_attempts":
            total_auth_attempts,

        "failed_auth_ratio":
            failed_auth_ratio,

        # ----------------------------------------------------
        # Identity behaviour
        # ----------------------------------------------------

        "unique_usernames":
            len(set(usernames)),

        "unique_passwords":
            len(set(passwords)),

        # ----------------------------------------------------
        # SSH fingerprint behaviour
        # ----------------------------------------------------

        "unique_hassh":
            len(set(hassh_values)),

        # ----------------------------------------------------
        # Protocol behaviour
        # ----------------------------------------------------

        "unique_protocols":
            len(set(protocols)),

        # ----------------------------------------------------
        # File behaviour
        # ----------------------------------------------------

        "num_file_events":
            len(file_events),

        # ----------------------------------------------------
        # Specific Cowrie event counts
        # ----------------------------------------------------

        "login_failed_events":
            event_counts.get(
                "cowrie.login.failed",
                0
            ),

        "login_success_events":
            event_counts.get(
                "cowrie.login.success",
                0
            ),

        "session_connect_events":
            event_counts.get(
                "cowrie.session.connect",
                0
            ),

        "session_closed_events":
            event_counts.get(
                "cowrie.session.closed",
                0
            ),

        "client_version_events":
            event_counts.get(
                "cowrie.client.version",
                0
            ),

        "client_kex_events":
            event_counts.get(
                "cowrie.client.kex",
                0
            ),

    }

    return features


# ============================================================
# MAIN DATA PROCESSING PIPELINE
# ============================================================

def main():

    print()
    print("=" * 60)
    print("ADAPTIVE DECEPTION")
    print("COWRIE FEATURE EXTRACTION")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Find all compressed JSON files
    # --------------------------------------------------------

    files = sorted(
        DATA_DIR.glob(
            "*.json.gz"
        )
    )

    print(
        f"Dataset files found: {len(files)}"
    )

    # --------------------------------------------------------
    # Check if files exist
    # --------------------------------------------------------

    if not files:

        print()
        print(
            "ERROR: No .json.gz files found."
        )

        print(
            f"Expected location: {DATA_DIR.resolve()}"
        )

        return

    # --------------------------------------------------------
    # Store all extracted features
    # --------------------------------------------------------

    all_features = {}

    # ========================================================
    # PROCESS EACH DATASET FILE
    # ========================================================

    for file_number, file_path in enumerate(
        files,
        start=1
    ):

        print()
        print(
            "-" * 60
        )

        print(
            f"[{file_number}/{len(files)}] "
            f"Processing: {file_path.name}"
        )

        print(
            "-" * 60
        )

        # ----------------------------------------------------
        # Load compressed JSON
        # ----------------------------------------------------

        data = load_file(
            file_path
        )

        print(
            f"Raw records: {len(data)}"
        )

        # ----------------------------------------------------
        # Reconstruct sessions
        # ----------------------------------------------------

        sessions = reconstruct_sessions(
            data
        )

        print(
            f"Sessions: {len(sessions)}"
        )

        # ----------------------------------------------------
        # Extract features from every session
        # ----------------------------------------------------

        for session_id, events in sessions.items():

            feature_record = (
                extract_session_features(
                    session_id,
                    events
                )
            )

            # ------------------------------------------------
            # If a session appears in multiple files,
            # append/replace carefully.
            # ------------------------------------------------

            if session_id not in all_features:

                all_features[
                    session_id
                ] = feature_record

            else:

                # This situation will be investigated later.
                # For now, keep the first complete record.
                pass

    # ========================================================
    # CONVERT TO LIST
    # ========================================================

    feature_records = list(
        all_features.values()
    )

    # ========================================================
    # OUTPUT FILE
    # ========================================================

    output_file = (
        OUTPUT_DIR /
        "session_features.json"
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            feature_records,
            file,
            indent=2
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Total sessions: {len(feature_records)}"
    )

    print()
    print(
        f"Output file:"
    )

    print(
        output_file.resolve()
    )

    # ========================================================
    # SHOW SAMPLE
    # ========================================================

    if feature_records:

        print()
        print("=" * 60)
        print("SAMPLE FEATURE RECORD")
        print("=" * 60)

        sample = feature_records[0]

        for key, value in sample.items():

            print(
                f"{key}: {value}"
            )

    print()
    print("=" * 60)
    print("NEXT STEP: DATA QUALITY INSPECTION")
    print("=" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()