#!/usr/bin/env python3
"""
Usage: python ci/generate_release_notes.py <previous_tag> <current_tag>

Generates release notes by comparing two git tags.

This uses the configuration in .github/release.yml to format the release notes.

Format for line is:

* <Title> by @<Author> in <PR Link>

Example output:

* fix: dir namespace cloud storage path removes one subdir level by @jackye1995 in https://github.com/lance-format/lance/pull/5495
* fix: panic unwrap on None in decoder.rs by @camilesing in https://github.com/lance-format/lance/pull/5424
* fix: ensure trailing slash is normalized in rest adapter by @jackye1995 in https://github.com/lance-format/lance/pull/5500

**Full Changelog**: https://github.com/lance-format/lance/compare/v1.0.0...v1.0.1
"""

import json
import re
import subprocess
import sys
from dataclasses import dataclass

import yaml

REPO = "lance-format/lance"
REPO_URL = f"https://github.com/{REPO}"


@dataclass
class Category:
    title: str
    labels: list[str]


@dataclass
class ChangelogConfig:
    exclude_labels: list[str]
    categories: list[Category]


@dataclass
class PullRequest:
    number: int
    title: str
    author: str
    labels: list[str]


def load_config(config_path: str = ".github/release.yml") -> ChangelogConfig:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    changelog = config.get("changelog", {})
    exclude_labels = changelog.get("exclude", {}).get("labels", [])

    categories = []
    for cat in changelog.get("categories", []):
        categories.append(Category(title=cat["title"], labels=cat["labels"]))

    return ChangelogConfig(exclude_labels=exclude_labels, categories=categories)


def get_commits_between_tags(previous_tag: str, current_tag: str) -> list[str]:
    """Get commit messages between two tags."""
    result = subprocess.run(
        ["git", "log", f"{previous_tag}..{current_tag}", "--format=%s"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().split("\n")


def extract_pr_number(commit_message: str) -> int | None:
    """Extract PR number from commit message like 'fix: something (#1234)'."""
    match = re.search(r"\(#(\d+)\)", commit_message)
    if match:
        return int(match.group(1))
    return None


def get_pr_details(pr_number: int) -> PullRequest | None:
    """Fetch PR details from GitHub API."""
    result = subprocess.run(
        [
            "gh",
            "pr",
            "view",
            str(pr_number),
            "--json",
            "title,author,labels",
            "--jq",
            "{title: .title, author: .author.login, labels: [.labels[].name]}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    data = json.loads(result.stdout)
    return PullRequest(
        number=pr_number,
        title=data["title"],
        author=data["author"],
        labels=data["labels"],
    )


def categorize_pr(pr: PullRequest, config: ChangelogConfig) -> str | None:
    """Return category title for a PR, or None if excluded."""
    # Check exclusions
    for label in pr.labels:
        if label in config.exclude_labels:
            return None

    # Find matching category
    for category in config.categories:
        if "*" in category.labels:
            return category.title
        for label in pr.labels:
            if label in category.labels:
                return category.title

    return None


def format_pr_entry(pr: PullRequest) -> str:
    """Format a single PR entry."""
    return f"* {pr.title} by @{pr.author} in {REPO_URL}/pull/{pr.number}"


def generate_release_notes(previous_tag: str, current_tag: str) -> str:
    config = load_config()
    commits = get_commits_between_tags(previous_tag, current_tag)

    # Collect unique PR numbers
    pr_numbers = set()
    for commit in commits:
        pr_num = extract_pr_number(commit)
        if pr_num:
            pr_numbers.add(pr_num)

    # Fetch PR details and categorize
    categorized: dict[str, list[PullRequest]] = {
        cat.title: [] for cat in config.categories
    }

    for pr_num in sorted(pr_numbers):
        pr = get_pr_details(pr_num)
        if pr is None:
            print(f"Warning: Could not fetch PR #{pr_num}", file=sys.stderr)
            continue

        category = categorize_pr(pr, config)
        if category:
            categorized[category].append(pr)

    # Build output
    lines = [
        f"<!-- Release notes generated using configuration in .github/release.yml at {current_tag} -->",
        "",
        "## What's Changed",
    ]

    for category in config.categories:
        prs = categorized[category.title]
        if prs:
            lines.append(f"### {category.title}")
            for pr in sorted(prs, key=lambda p: p.number):
                lines.append(format_pr_entry(pr))

    lines.append(
        f"\n**Full Changelog**: {REPO_URL}/compare/{previous_tag}...{current_tag}"
    )

    return "\n".join(lines)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    previous_tag = sys.argv[1]
    current_tag = sys.argv[2]

    notes = generate_release_notes(previous_tag, current_tag)
    print(notes)


if __name__ == "__main__":
    main()
