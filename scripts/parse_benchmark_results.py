#!/usr/bin/env python
"""Parse the benchmark results and generate a json file for uploading/plotting"""
import json
import os
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
        duration_ns = estimates["Mean"]["point_estimate"]
    return {
        "name": name,
        "commit": commit_sha,
        "datetime": datetime_str,
        "os": os,
        "arch": arch,
        "duration_ns": duration_ns,
    }


def parse_all(root=ROOT):
    """Parse all the benchmark results"""
    results = []
    for path in root.iterdir():
        if path.is_dir():
            results.append(parse_one(path))
    print(results)
    return results