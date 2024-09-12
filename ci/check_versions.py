"""
This validates that we have incremented the version correctly in the presence
of breaking changes.
"""

from github import Github
import os


def get_versions():
    """
    Gets the current version in both python/Cargo.toml and Cargo.toml files.
    """
    import tomllib

    with open("python/Cargo.toml", "rb") as file:
        pylance_version = tomllib.load(file)["package"]["version"]
    with open("Cargo.toml", "rb") as file:
        rust_data = tomllib.load(file)
    rust_version = rust_data["workspace"]["package"]["version"]
    package_versions = {
        name: data["version"].strip("=")
        for name, data in rust_data["workspace"]["dependencies"].items()
        if isinstance(data, dict) and "version" in data and name.startswith("lance-")
    }

    assert pylance_version == rust_version
    for name, version in package_versions.items():
        assert version == pylance_version, (name, version, pylance_version)

    return pylance_version


def parse_version(version: str) -> tuple[int, int, int]:
    return tuple(map(int, version.split(".")))


if __name__ == "__main__":
    new_version = parse_version(get_versions())
    print(f"New version: {new_version}")

    repo = Github().get_repo(os.environ["GITHUB_REPOSITORY"])
    latest_release = repo.get_latest_release()
    last_version = parse_version(latest_release.tag_name[1:])
    print(f"Last version: {last_version}")

    # Check for a breaking-change label in the PRs between the last release and the current commit.
    commits = repo.compare(latest_release.tag_name, os.environ["GITHUB_SHA"]).commits
    prs = (pr for commit in commits for pr in commit.get_pulls())
    has_breaking_changes = False
    for pr in prs:
        pr_labels = (label.name for label in pr.labels)
        if any(label == "breaking-change" for label in pr_labels):
            has_breaking_changes = True
            print(f"Found breaking change in PR #{pr.number}: {pr.title}")
            print(f"  {pr.html_url}")
            break
    else:
        print("No breaking changes found.")

    if os.environ.get("PR_NUMBER"):
        # If we're running on a PR, we should validate that the version has been
        # bumped correctly.
        pr_number = int(os.environ["PR_NUMBER"])
        pr = repo.get_pull(pr_number)
        pr_labels = [label.name for label in pr.get_labels()]
        has_breaking_changes = "breaking-change" in pr_labels or has_breaking_changes

    if has_breaking_changes:
        # Minor version needs to have been bumped.
        assert (
            new_version[1] == last_version[1] + 1
        ), "Minor version should have been bumped because there was a breaking change."
