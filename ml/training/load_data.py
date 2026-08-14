import gzip
import json
from pathlib import Path


DATA_DIR = Path("data")

files = sorted(DATA_DIR.glob("*.json.gz"))

print("Files found:", len(files))

for file_path in files:

    print("\n==============================")
    print("FILE:", file_path.name)
    print("==============================")

    with gzip.open(file_path, "rt", encoding="utf-8") as file:
        data = json.load(file)

    print("Python type:", type(data))

    if isinstance(data, list):

        print("Number of records:", len(data))

        if len(data) > 0:
            print("First record:")
            print(data[0])

    elif isinstance(data, dict):

        print("Dictionary keys:")
        print(data.keys())

    else:

        print("Unknown data structure")