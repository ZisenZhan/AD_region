"""Verify the archived result files using only the Python standard library."""

import csv
import hashlib
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    failures = []
    total_bytes = 0
    count = 0
    with (root / "DATA_MANIFEST.csv").open(encoding="utf-8-sig", newline="") as stream:
        entries = list(csv.DictReader(stream))
    seen = set()
    for entry in entries:
        relative = entry["path"]
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or relative in seen:
            failures.append(f"Invalid or duplicate path: {relative}")
            continue
        seen.add(relative)
        if not path.is_file():
            failures.append(f"Missing file: {relative}")
            continue
        size = path.stat().st_size
        result = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                result.update(block)
        if size != int(entry["bytes"]) or result.hexdigest() != entry["sha256"]:
            failures.append(f"Size or SHA256 mismatch: {relative}")
        if entry["transformation"] == "none" and entry["sha256"] != entry["source_sha256"]:
            failures.append(f"Unexpected source difference: {relative}")
        if size >= 100 * 1024 * 1024:
            failures.append(f"File exceeds ordinary Git upload limit: {relative}")
        total_bytes += size
        count += 1
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    print(f"Verified {count} archived result files ({total_bytes / 1024**2:.2f} MiB).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
