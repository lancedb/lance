#!/usr/bin/env python3

from __future__ import annotations

import io
import json
import subprocess
import sys
import unittest
from unittest import mock
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import environment_attestation as attestation  # noqa: E402


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = io.BytesIO(payload)

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload.read()


class EnvironmentAttestationTests(unittest.TestCase):
    def test_attests_matching_ec2_and_bucket_regions(self) -> None:
        identity = json.dumps(
            {
                "instanceId": "i-0123456789abcdef0",
                "availabilityZone": "us-east-2b",
                "region": "us-east-2",
            }
        ).encode()
        with (
            mock.patch.object(
                attestation.urllib.request,
                "urlopen",
                side_effect=[_Response(b"token"), _Response(identity)],
            ),
            mock.patch.object(
                attestation.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, "us-east-2\n"),
            ),
        ):
            value = attestation.attest_same_region_s3("s3://release-bucket/prefix")
        self.assertEqual(value["compute_region"], "us-east-2")
        self.assertEqual(value["bucket_region"], "us-east-2")
        self.assertEqual(value["bucket"], "release-bucket")
        self.assertIs(
            attestation.validate_same_region_s3_attestation(
                value, "s3://release-bucket/prefix"
            ),
            value,
        )

    def test_rejects_cross_region_release_storage(self) -> None:
        with (
            mock.patch.object(
                attestation,
                "ec2_identity_document",
                return_value={
                    "instance_id": "i-1",
                    "availability_zone": "us-east-2a",
                    "compute_region": "us-east-2",
                },
            ),
            mock.patch.object(
                attestation, "s3_bucket_region", return_value="us-west-2"
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                attestation.attest_same_region_s3("s3://release-bucket/prefix")

    def test_normalizes_legacy_bucket_location_values(self) -> None:
        for raw, expected in (("None\n", "us-east-1"), ("EU\n", "eu-west-1")):
            with (
                self.subTest(raw=raw),
                mock.patch.object(
                    attestation.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 0, raw),
                ),
            ):
                self.assertEqual(
                    attestation.s3_bucket_region("release-bucket"), expected
                )

    def test_validation_binds_attestation_to_dataset_bucket(self) -> None:
        value = {
            "schema_version": 1,
            "method": attestation.ATTESTATION_METHOD,
            "instance_id": "i-1",
            "availability_zone": "us-east-2a",
            "compute_region": "us-east-2",
            "bucket": "release-bucket",
            "bucket_region": "us-east-2",
        }
        with self.assertRaisesRegex(ValueError, "wrong bucket"):
            attestation.validate_same_region_s3_attestation(
                value, "s3://other-bucket/prefix"
            )


if __name__ == "__main__":
    unittest.main()
