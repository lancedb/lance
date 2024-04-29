"""Add pre-release version to the package."""

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
                lines[i] = f'version = "{args.version}"\n'
                break
        else:
            raise ValueError("Could not find version in Cargo.toml")

    parsed_version = parse(args.version)
    current_version = parse(current_version)

    assert (
        parsed_version.release == current_version.release
    ), f"Base versions do not match: {parsed_version.release} != {current_version.release}"

    with open("python/Cargo.toml", "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
