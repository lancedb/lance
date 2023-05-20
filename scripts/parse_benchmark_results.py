#!/usr/bin/env python3
"""Parse the benchmark results and generate a json file for uploading/plotting"""
import json
from pathlib import Path
import platform
import subprocess
import datetime


ROOT = Path(__file__).parent.parent / "rust" / "target" / "criterion"


def get_commit_sha():
    """Get the commit sha"""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


def get_current_datetime():
    """Get the current datetime"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_os():
    """Get the operating system"""
    return platform.system()


def get_arch():
    """Get the architecture"""
    return platform.machine()


def parse_one(path: Path):
    """Parse one benchmark result"""
    name = path.name
    commit_sha = get_commit_sha()
    datetime_str = get_current_datetime()
    os = get_os()
    arch = get_arch()
    with (path / "new" / "estimates.json").open() as f:
        estimates = json.load(f)
        duration_ns = estimates["mean"]["point_estimate"]
        standard_error = estimates["mean"]["standard_error"]
    return {
        "name": name,
        "commit": commit_sha,
        "datetime": datetime_str,
        "os": os,
        "arch": arch,
        "mean_duration_ns": duration_ns,
        "standard_error": standard_error
    }


def parse_all(root=ROOT):
    """Parse all the benchmark results"""
    results = []
    for path in root.iterdir():
        if path.is_dir():
            results.append(parse_one(path))
    return results


if __name__ == "__main__":
    rs = parse_all()
    commit = get_commit_sha()
    with (Path(__file__).parent.parent / f"results_{commit}.json").open("w") as f:
        json.dump(rs, f, indent=2)
