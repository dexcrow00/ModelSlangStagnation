import csv

from ccdiff.diff import DiffRecord


def write_csv(records: list[DiffRecord], path: str) -> None:
    """Write diff records to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["url", "status", "old_content", "new_content", "diff"])
        for r in records:
            writer.writerow([r.url, r.status, r.old_content, r.new_content, r.diff_summary])
