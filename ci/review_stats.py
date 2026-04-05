#!/usr/bin/env python3
"""
Usage: python ci/review_stats.py

Counts code reviews by contributors in the last 30 days.
Users without write permission are marked with an asterisk (*).
"""

import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

REPO = "lance-format/lance"
CACHE_DIR = Path.home() / ".cache" / "lance"
CACHE_FILE = CACHE_DIR / "review_stats_cache.json"


def load_cache() -> dict:
    """Load cached data."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_cache(cache: dict) -> None:
    """Save data to cache."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except OSError as e:
        print(f"Warning: Could not save cache: {e}", file=sys.stderr)


def get_prs_last_30_days() -> list[int]:
    """Get all merged PR numbers from the last 30 days using search API."""
    cutoff_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    pr_numbers = []
    page = 1
    per_page = 100

    print("Fetching PRs from the last 30 days...", file=sys.stderr)
    while True:
        # Use search API for better date filtering
        query = f"repo:{REPO} is:pr is:merged merged:>={cutoff_date}"
        result = subprocess.run(
            [
                "gh",
                "api",
                "search/issues",
                "-X",
                "GET",
                "-f",
                f"q={query}",
                "-f",
                f"per_page={per_page}",
                "-f",
                f"page={page}",
                "--jq",
                ".items[].number",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            break

        numbers = [int(n) for n in result.stdout.strip().split("\n") if n]
        if not numbers:
            break

        pr_numbers.extend(numbers)
        print(
            f"  Fetched page {page} ({len(pr_numbers)} PRs so far)...", file=sys.stderr
        )

        if len(numbers) < per_page:
            break
        page += 1

    return pr_numbers


def get_reviews_for_pr(pr_number: int) -> list[str]:
    """Get list of reviewer usernames for a PR."""
    result = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{REPO}/pulls/{pr_number}/reviews",
            "--jq",
            ".[].user.login",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []

    return [u for u in result.stdout.strip().split("\n") if u]


def get_collaborators_with_write() -> set[str]:
    """Get set of GitHub usernames with write permission."""
    collaborators = set()

    result = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{REPO}/collaborators",
            "--paginate",
            "--jq",
            ".[] | select(.permissions.push == true or .permissions.admin == true or .permissions.maintain == true) | .login",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        for line in result.stdout.strip().split("\n"):
            if line:
                collaborators.add(line.lower())

    return collaborators


def main():
    cache = load_cache()

    # Get PRs from the last 30 days
    pr_numbers = get_prs_last_30_days()
    print(f"Found {len(pr_numbers)} merged PRs", file=sys.stderr)

    # Get reviews for each PR
    print("Fetching reviews...", file=sys.stderr)
    review_counts: dict[str, int] = defaultdict(int)
    cached_prs = cache.get("pr_reviews", {})
    uncached_count = 0

    for i, pr_num in enumerate(pr_numbers):
        pr_key = str(pr_num)
        if pr_key in cached_prs:
            reviewers = cached_prs[pr_key]
        else:
            reviewers = get_reviews_for_pr(pr_num)
            cached_prs[pr_key] = reviewers
            uncached_count += 1

        for reviewer in reviewers:
            review_counts[reviewer.lower()] += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(pr_numbers)} PRs...", file=sys.stderr)

    cache["pr_reviews"] = cached_prs
    save_cache(cache)

    if uncached_count > 0:
        print(
            f"  Fetched {uncached_count} PRs via API, {len(pr_numbers) - uncached_count} from cache",
            file=sys.stderr,
        )

    # Get collaborators with write permission
    print("Fetching repository collaborators...", file=sys.stderr)
    write_collaborators = get_collaborators_with_write()
    print(
        f"Found {len(write_collaborators)} collaborators with write access",
        file=sys.stderr,
    )

    # Build results list
    reviewers = []
    for username, count in review_counts.items():
        has_write = username in write_collaborators
        reviewers.append(
            {
                "username": username,
                "reviews": count,
                "has_write": has_write,
            }
        )

    # Sort by review count descending
    reviewers.sort(key=lambda x: x["reviews"], reverse=True)

    # Print results
    print("\nCode reviews by contributor (sorted by review count):\n")
    print("* = no write permission\n")
    print(f"{'Reviews':<10} {'Username':<30}")
    print("-" * 40)
    for reviewer in reviewers:
        marker = "" if reviewer["has_write"] else "*"
        print(f"{reviewer['reviews']:<10} {reviewer['username']:<30} {marker}")

    total_with_write = sum(1 for r in reviewers if r["has_write"])
    total_without_write = len(reviewers) - total_with_write
    print(
        f"\nTotal: {len(reviewers)} reviewers ({total_with_write} with write access, {total_without_write} without)"
    )


if __name__ == "__main__":
    main()
