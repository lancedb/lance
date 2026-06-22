"""Regenerate the PMC roster table in docs from the machine-readable roster.

`docs/src/community/pmc.yaml` is the source of truth for the PMC roster (it also
drives the format-specification vote gate). This script renders that data into
the roster table in `docs/src/community/pmc.md`, between the
`<!-- BEGIN PMC ROSTER -->` and `<!-- END PMC ROSTER -->` markers.

Usage:
    python ci/sync_pmc_docs.py            # rewrite pmc.md in place
    python ci/sync_pmc_docs.py --check    # exit 1 if pmc.md is out of sync
"""

import argparse
import pathlib
import sys

import yaml

COMMUNITY_DIR = pathlib.Path(__file__).resolve().parent.parent / "docs/src/community"
ROSTER_YAML = COMMUNITY_DIR / "pmc.yaml"
ROSTER_MD = COMMUNITY_DIR / "pmc.md"

BEGIN_MARKER = "<!-- BEGIN PMC ROSTER -->"
END_MARKER = "<!-- END PMC ROSTER -->"

COLUMNS = [
    ("Name", "name"),
    ("GitHub Handle", "handle"),
    ("Affiliation", "affiliation"),
    ("Ecosystem Roles", "ecosystem_roles"),
]


def render_table(members):
    rows = [
        [str(member.get(key, "") or "") for _, key in COLUMNS] for member in members
    ]
    headers = [title for title, _ in COLUMNS]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        if rows
        else len(headers[i])
        for i in range(len(COLUMNS))
    ]

    def format_row(cells):
        return (
            "| "
            + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))
            + " |"
        )

    lines = [format_row(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def build_markdown():
    data = yaml.safe_load(ROSTER_YAML.read_text())
    members = data["members"]
    table = render_table(members)
    block = f"{BEGIN_MARKER}\n{table}\n{END_MARKER}"

    original = ROSTER_MD.read_text()
    start = original.find(BEGIN_MARKER)
    end = original.find(END_MARKER)
    if start == -1 or end == -1:
        raise SystemExit(
            f"Could not find roster markers in {ROSTER_MD}. Expected "
            f"{BEGIN_MARKER!r} and {END_MARKER!r}."
        )
    return original[:start] + block + original[end + len(END_MARKER) :]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if pmc.md is out of sync instead of rewriting it.",
    )
    args = parser.parse_args()

    updated = build_markdown()
    if args.check:
        if ROSTER_MD.read_text() != updated:
            print(
                f"{ROSTER_MD} is out of sync with {ROSTER_YAML}. "
                "Run `python ci/sync_pmc_docs.py` and commit the result.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"{ROSTER_MD.name} is in sync with {ROSTER_YAML.name}.")
    else:
        ROSTER_MD.write_text(updated)
        print(f"Wrote roster table to {ROSTER_MD}.")


if __name__ == "__main__":
    main()
