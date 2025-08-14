#!/usr/bin/env python3
"""
Check for breaking changes by examining GitHub PR labels.

This script is used during the release process to ensure we don't accidentally 
release breaking changes as a patch version.
"""

import argparse
import sys
import os
from github import Github


def check_github_pr_labels() -> bool:
    """Check for breaking-change labels in PRs between last release and current commit."""
    # Require GitHub environment variables
    if not os.environ.get("GITHUB_REPOSITORY"):
        print("Error: GITHUB_REPOSITORY environment variable not set")
        sys.exit(1)
    
    try:
        # Initialize GitHub client
        token = os.environ.get("GITHUB_TOKEN")
        g = Github(token) if token else Github()
        repo = g.get_repo(os.environ["GITHUB_REPOSITORY"])
        
        # Get the latest release
        try:
            latest_release = repo.get_latest_release()
            last_tag = latest_release.tag_name
        except:
            print("No previous releases found, skipping breaking change check")
            return False
        
        print(f"Checking for breaking changes since {last_tag}")
        
        # Get commits between last release and current SHA
        sha = os.environ.get("GITHUB_SHA", "HEAD")
        comparison = repo.compare(last_tag, sha)
        
        # Check all PRs for breaking-change label
        breaking_prs = []
        checked_prs = set()
        
        for commit in comparison.commits:
            # Get PRs associated with this commit
            prs = list(commit.get_pulls())
            
            for pr in prs:
                # Skip if we've already checked this PR
                if pr.number in checked_prs:
                    continue
                checked_prs.add(pr.number)
                
                # Check for breaking-change label
                pr_labels = [label.name for label in pr.labels]
                if "breaking-change" in pr_labels:
                    breaking_prs.append(pr)
                    print(f"  Found breaking change in PR #{pr.number}: {pr.title}")
                    print(f"    {pr.html_url}")
        
        if breaking_prs:
            return True
        else:
            print("  No breaking changes found in PR labels")
            return False
                
    except Exception as e:
        print(f"Error checking GitHub PR labels: {e}")
        # If we can't check, assume no breaking changes to avoid blocking releases
        print("Warning: Could not verify breaking changes, proceeding anyway")
        return False


def main():
    """Main function to check for breaking changes."""
    parser = argparse.ArgumentParser(
        description="Check for breaking changes and validate release type"
    )
    parser.add_argument(
        "--release-type",
        choices=["patch", "minor", "major"],
        required=True,
        help="Type of release being performed"
    )
    args = parser.parse_args()
    
    print(f"Checking for breaking changes (Release type: {args.release_type})...")
    print("-" * 50)
    
    has_breaking_changes = check_github_pr_labels()
    
    print("-" * 50)
    
    if has_breaking_changes:
        if args.release_type == "patch":
            print("✗ Breaking changes detected but patch release requested!")
            print("Please use 'minor' or 'major' version bump for the release.")
            sys.exit(1)
        else:
            print(f"⚠️ Breaking changes detected, proceeding with {args.release_type} release")
            print("This is allowed since you're using a minor or major version bump.")
            sys.exit(0)
    else:
        print("✓ No breaking changes detected")
        print(f"Proceeding with {args.release_type} release")
        sys.exit(0)


if __name__ == "__main__":
    main()