#!/usr/bin/env python3
"""
Usage: python ci/new_contributors.py

Counts commits by authors in the last year, including co-authors.
Only shows contributors without write permission to the repository.
"""

import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

REPO = "lance-format/lance"
CACHE_DIR = Path.home() / ".cache" / "lance"
CACHE_FILE = CACHE_DIR / "contributor_stats_cache.json"

# Emails to exclude from results (bots, automated accounts, etc.)
EXCLUDED_EMAILS = {
    "noreply@anthropic.com",
    "lance-dev@lancedb.com",
    "dev+gha@lance.org",
}


def load_username_cache() -> dict[str, str]:
    """Load cached email -> username mappings."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_username_cache(cache: dict[str, str]) -> None:
    """Save email -> username mappings to cache."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except OSError as e:
        print(f"Warning: Could not save cache: {e}", file=sys.stderr)


def get_commits_last_year() -> list[dict]:
    """Get all commits from the last year with author and body."""
    one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    result = subprocess.run(
        [
            "git",
            "log",
            f"--since={one_year_ago}",
            "--format=%H%x00%an%x00%ae%x00%b%x00%x01",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    commits = []
    for entry in result.stdout.split("\x01"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split("\x00")
        if len(parts) >= 4:
            commits.append(
                {
                    "hash": parts[0],
                    "author_name": parts[1],
                    "author_email": parts[2],
                    "body": parts[3],
                }
            )
    return commits


def extract_co_authors(body: str) -> list[tuple[str, str]]:
    """Extract co-authors from commit body.

    Returns list of (name, email) tuples.
    """
    co_authors = []
    pattern = r"Co-authored-by:\s*(.+?)\s*<([^>]+)>"
    for match in re.finditer(pattern, body, re.IGNORECASE):
        co_authors.append((match.group(1).strip(), match.group(2).strip()))
    return co_authors


def get_github_username_from_email(email: str) -> str | None:
    """Try to get GitHub username from email pattern."""
    # Handle GitHub noreply emails
    match = re.match(r"(\d+\+)?([^@]+)@users\.noreply\.github\.com", email)
    if match:
        return match.group(2)
    return None


def resolve_usernames_via_api(
    email_to_sample_commit: dict[str, str],
    cache: dict[str, str],
) -> dict[str, str]:
    """Get GitHub usernames by querying one sample commit per email.

    Uses cache for previously resolved emails.
    Returns mapping of email -> GitHub username.
    """
    email_to_username: dict[str, str] = {}

    # Check cache first
    uncached_emails = {}
    for email, sha in email_to_sample_commit.items():
        if email in cache:
            email_to_username[email] = cache[email]
        else:
            uncached_emails[email] = sha

    if uncached_emails:
        items = list(uncached_emails.items())
        total = len(items)
        cached_count = len(email_to_sample_commit) - total
        if cached_count > 0:
            print(
                f"  Found {cached_count} cached, resolving {total} via API...",
                file=sys.stderr,
            )

        for i, (email, sha) in enumerate(items):
            result = subprocess.run(
                [
                    "gh",
                    "api",
                    f"repos/{REPO}/commits/{sha}",
                    "--jq",
                    ".author.login // empty",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                username = result.stdout.strip()
                if username:
                    email_to_username[email] = username
                    cache[email] = username

            if (i + 1) % 20 == 0:
                print(f"  Resolved {i + 1}/{total} authors...", file=sys.stderr)

    return email_to_username


def get_collaborators_with_write() -> set[str]:
    """Get set of GitHub usernames with write permission."""
    collaborators = set()

    # Get collaborators with push/admin/maintain permission
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
    print("Fetching commits from the last year...", file=sys.stderr)
    commits = get_commits_last_year()
    print(f"Found {len(commits)} commits", file=sys.stderr)

    # Load username cache
    username_cache = load_username_cache()

    # Count commits per author (by email, since that's more reliable)
    # Also track one sample commit hash per email for API resolution
    author_commits: dict[str, int] = defaultdict(int)
    email_to_name: dict[str, str] = {}
    email_to_sample_commit: dict[str, str] = {}

    for commit in commits:
        # Add main author
        email = commit["author_email"].lower()
        author_commits[email] += 1
        email_to_name[email] = commit["author_name"]
        if email not in email_to_sample_commit:
            email_to_sample_commit[email] = commit["hash"]

        # Add co-authors (they don't have commit hashes, so we can't resolve via API)
        for name, co_email in extract_co_authors(commit["body"]):
            co_email = co_email.lower()
            author_commits[co_email] += 1
            if co_email not in email_to_name:
                email_to_name[co_email] = name

    print(f"Found {len(author_commits)} unique authors", file=sys.stderr)

    # First pass: get usernames from email patterns (noreply emails)
    email_to_username: dict[str, str | None] = {}
    for email in author_commits:
        email_to_username[email] = get_github_username_from_email(email)

    # Second pass: resolve remaining emails via GitHub API (only for main authors)
    emails_to_resolve = {
        email: sha
        for email, sha in email_to_sample_commit.items()
        if email_to_username.get(email) is None
    }
    print(
        f"Resolving {len(emails_to_resolve)} usernames via GitHub API...",
        file=sys.stderr,
    )
    api_mappings = resolve_usernames_via_api(emails_to_resolve, username_cache)
    for email, username in api_mappings.items():
        email_to_username[email] = username

    # Save updated cache
    save_username_cache(username_cache)

    # Get collaborators with write permission
    print("Fetching repository collaborators...", file=sys.stderr)
    write_collaborators = get_collaborators_with_write()
    print(
        f"Found {len(write_collaborators)} collaborators with write access",
        file=sys.stderr,
    )

    # Filter to only non-write contributors
    non_write_contributors = []
    for email, count in author_commits.items():
        # Skip excluded emails
        if email in EXCLUDED_EMAILS:
            continue
        username = email_to_username.get(email)
        if username and username.lower() in write_collaborators:
            continue
        # Include if we couldn't determine username or if they don't have write access
        non_write_contributors.append(
            {
                "email": email,
                "name": email_to_name[email],
                "username": username,
                "commits": count,
            }
        )

    # Sort by commit count descending
    non_write_contributors.sort(key=lambda x: x["commits"], reverse=True)

    # Print results
    print("\nContributors without write permission (sorted by commit count):\n")
    print(f"{'Commits':<10} {'Username':<25} {'Name':<30} {'Email'}")
    print("-" * 100)
    for contributor in non_write_contributors:
        username = contributor["username"] or "(unknown)"
        print(
            f"{contributor['commits']:<10} {username:<25} {contributor['name']:<30} {contributor['email']}"
        )

    print(f"\nTotal: {len(non_write_contributors)} contributors without write access")


if __name__ == "__main__":
    main()
