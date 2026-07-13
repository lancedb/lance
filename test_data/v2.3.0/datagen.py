#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Regenerate the Lance 2.3 sparse-reader compatibility fixture."""

import json
from pathlib import Path
import struct
import subprocess


FIXTURE_DIR = Path(__file__).resolve().parent
MANIFEST = FIXTURE_DIR / "datagen" / "Cargo.toml"
EXPECTED_LANCE_COMMIT = "1aca0a6d4fbb1010adb3b8fc2d0b6951a11e736b"
EXPECTED_LANCE_PACKAGE_VERSION = "9.0.0-beta.12"
EXPECTED_FILE_VERSION = (2, 3)


metadata = json.loads(
    subprocess.check_output(
        [
            "cargo",
            "metadata",
            "--format-version",
            "1",
            "--manifest-path",
            str(MANIFEST),
        ],
        text=True,
    )
)
lance_file = next(package for package in metadata["packages"] if package["name"] == "lance-file")
assert lance_file["version"] == EXPECTED_LANCE_PACKAGE_VERSION, (
    f"Expected lance-file {EXPECTED_LANCE_PACKAGE_VERSION}, got {lance_file['version']}"
)
assert EXPECTED_LANCE_COMMIT in lance_file["source"], (
    f"Expected Lance prototype commit {EXPECTED_LANCE_COMMIT}, got {lance_file['source']}"
)

subprocess.run(
    [
        "cargo",
        "run",
        "--manifest-path",
        str(MANIFEST),
        "--",
        str(FIXTURE_DIR),
    ],
    check=True,
)

for fixture_name in ("sparse_reader", "empty_sparse_reader"):
    lance_path = FIXTURE_DIR / f"{fixture_name}.lance"
    arrow_path = FIXTURE_DIR / f"{fixture_name}.arrow"
    assert lance_path.is_file(), f"Generator did not create {lance_path.name}"
    assert arrow_path.is_file(), f"Generator did not create {arrow_path.name}"
    footer = lance_path.read_bytes()[-8:]
    major, minor = struct.unpack("<HH", footer[:4])
    assert (major, minor) == EXPECTED_FILE_VERSION, (
        f"Expected Lance file version {EXPECTED_FILE_VERSION}, got {(major, minor)}"
    )
    assert footer[4:] == b"LANC", f"{lance_path.name} has invalid Lance magic"
