#!/usr/bin/env python3
"""
Version management script for Lance project.
Handles version bumping across all project components.
"""

import argparse
import subprocess
import sys
import json
import re
from pathlib import Path
from typing import Tuple, Optional


def run_command(cmd: list[str], capture_output: bool = True, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=capture_output, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Error running command: {' '.join(cmd)}")
        if capture_output:
            print(f"stderr: {result.stderr}")
        sys.exit(result.returncode)
    return result


def get_current_version() -> str:
    """Get the current version from Cargo.toml."""
    cargo_toml = Path("Cargo.toml")
    with open(cargo_toml, "r") as f:
        for line in f:
            if line.strip().startswith('version = "'):
                return line.split('"')[1]
    raise ValueError("Could not find version in Cargo.toml")


def parse_version(version: str) -> Tuple[int, int, int]:
    """Parse a version string into major, minor, patch components."""
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        raise ValueError(f"Invalid version format: {version}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def bump_version(current: str, bump_type: str) -> str:
    """Calculate the new version based on bump type."""
    major, minor, patch = parse_version(current)
    
    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")


def update_cargo_lock_files():
    """Update all Cargo.lock files after version change."""
    lock_files = [
        "Cargo.lock",
        "python/Cargo.lock",
        "java/lance-jni/Cargo.lock",
    ]
    
    for lock_file in lock_files:
        if Path(lock_file).exists():
            directory = Path(lock_file).parent
            print(f"Updating {lock_file}...")
            run_command(["cargo", "update", "-p", "lance"], cwd=directory if directory != Path(".") else None)


def validate_version_consistency():
    """Validate that all versions are consistent across the project."""
    version = get_current_version()
    errors = []
    
    # Check all creates with explicit versioning
    rust_crates = [
        "python/Cargo.toml",
        "java/lance-jni/Cargo.toml",
    ]
    
    for crate_path in rust_crates:
        if Path(crate_path).exists():
            with open(crate_path, "r") as f:
                content = f.read()
                if f'version = "{version}"' not in content:
                    errors.append(f"{crate_path} has inconsistent version")
    
    if errors:
        print("Version consistency check failed:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print(f"All components are at version {version}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Bump Lance project version")
    parser.add_argument(
        "bump_type",
        choices=["major", "minor", "patch"],
        help="Type of version bump to perform"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip version consistency validation"
    )
    
    args = parser.parse_args()
    
    # Get current version
    current_version = get_current_version()
    new_version = bump_version(current_version, args.bump_type)
    
    print(f"Current version: {current_version}")
    print(f"New version: {new_version}")
    
    if args.dry_run:
        print("Dry run - no changes made")
        return
    
    # Use bump-my-version to update all files
    print("\nUpdating version in all files...")
    run_command(["bump-my-version", "bump", "--current-version", current_version, "--new-version", new_version, "--ignore-missing-version", "--ignore-missing-files"])
    
    # Update Cargo.lock files
    print("\nUpdating Cargo.lock files...")
    update_cargo_lock_files()
    
    # Validate consistency
    if not args.no_validate:
        print("\nValidating version consistency...")
        if not validate_version_consistency():
            print("Version update may have failed. Please check manually.")
            sys.exit(1)
    
    print(f"\nSuccessfully bumped version from {current_version} to {new_version}")


if __name__ == "__main__":
    main()