"""Add pre-release version to the package.

The version of the Python package is always checked in as the stable version
(e.g. 0.10.17). This script is used to set the pre-release version
(e.g. 0.10.17-beta.1).
"""

from packaging.version import parse
import argparse


def main():
    parser = argparse.ArgumentParser(description="Set the version of the package.")
    parser.add_argument("version", type=str, help="The version to set.")
    args = parser.parse_args()

    with open("python/Cargo.toml", "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("version = "):
                current_version = line.split('"')[1]
                # Remove the v prefix
                lines[i] = f'version = "{args.version[1:]}"\n'
                break
        else:
            raise ValueError("Could not find version in Cargo.toml")

    parsed_version = parse(args.version)
    current_version = parse(current_version)

    # Make sure the release versions match (excludes pre-release specifiers). That is,
    # Allow 0.10.17 to be updated to 0.10.17-beta.1
    # Disallow 0.10.17 to be updated to 0.11.0-beta.1
    assert (
        parsed_version.release == current_version.release
    ), f"Base versions do not match: {parsed_version.release} != {current_version.release}"

    with open("python/Cargo.toml", "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
