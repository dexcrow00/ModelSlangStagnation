import difflib
from dataclasses import dataclass


@dataclass
class DiffRecord:
    url: str
    status: str  # "added", "removed", or "changed"
    old_content: str
    new_content: str
    diff_summary: str


def compute_diff(old: dict[str, str], new: dict[str, str]) -> list[DiffRecord]:
    """Compare two URL→content maps and produce a list of DiffRecords."""
    records: list[DiffRecord] = []
    all_urls = sorted(set(old) | set(new))

    for url in all_urls:
        in_old = url in old
        in_new = url in new

        if in_old and not in_new:
            records.append(DiffRecord(url, "removed", old[url], "", ""))
        elif in_new and not in_old:
            records.append(DiffRecord(url, "added", "", new[url], ""))
        else:
            old_text = old[url]
            new_text = new[url]
            if old_text != new_text:
                diff = "\n".join(
                    difflib.unified_diff(
                        old_text.splitlines(),
                        new_text.splitlines(),
                        lineterm="",
                        fromfile="old",
                        tofile="new",
                    )
                )
                records.append(DiffRecord(url, "changed", old_text, new_text, diff))

    return records
