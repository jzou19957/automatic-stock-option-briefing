import os
import sys
import requests

CSV_URL = os.getenv("UNIVERSE_CSV_URL", "").strip()


def main() -> None:
    if not CSV_URL:
        print("ERROR: UNIVERSE_CSV_URL is missing")
        sys.exit(1)

    response = requests.get(CSV_URL, timeout=30)
    response.raise_for_status()

    text = response.text.strip()
    if not text:
        print("ERROR: Downloaded CSV is empty")
        sys.exit(1)

    with open("universe.csv", "w", encoding="utf-8", newline="") as f:
        f.write(text)

    print("Saved universe.csv")
    print("Header:", text.splitlines()[0])


if __name__ == "__main__":
    main()
