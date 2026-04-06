import argparse
import sys

from ccdiff.warc_reader import read_warc
from ccdiff.diff import compute_diff
from ccdiff.output import write_csv


def main():
    parser = argparse.ArgumentParser(
        prog="ccdiff",
        description="Diff two Common Crawl WARC archives and output changes as CSV.",
    )
    parser.add_argument("warc_old", help="Path to the older WARC file")
    parser.add_argument("warc_new", help="Path to the newer WARC file")
    parser.add_argument(
        "-o", "--output", default="diff.csv", help="Output CSV path (default: diff.csv)"
    )
    args = parser.parse_args()

    print(f"Reading old archive: {args.warc_old}")
    old = read_warc(args.warc_old)
    print(f"  → {len(old)} URLs")

    print(f"Reading new archive: {args.warc_new}")
    new = read_warc(args.warc_new)
    print(f"  → {len(new)} URLs")

    print("Computing diff…")
    records = compute_diff(old, new)

    added = sum(1 for r in records if r.status == "added")
    removed = sum(1 for r in records if r.status == "removed")
    changed = sum(1 for r in records if r.status == "changed")
    print(f"  → {added} added, {removed} removed, {changed} changed")

    write_csv(records, args.output)
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
