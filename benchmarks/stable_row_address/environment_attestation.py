#!/usr/bin/env python3
"""Attest that a release benchmark host and S3 bucket share one region."""

from __future__ import annotations

import json
import subprocess
import urllib.request
from typing import Any
from urllib.parse import urlsplit


IMDS_BASE_URL = "http://169.254.169.254/latest"
ATTESTATION_METHOD = "ec2-imdsv2+aws-s3api-get-bucket-location"
ATTESTATION_FIELDS = {
    "schema_version",
    "method",
    "instance_id",
    "availability_zone",
    "compute_region",
    "bucket",
    "bucket_region",
}


def _non_empty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value.strip()


def s3_bucket(dataset_root: str) -> str:
    parsed = urlsplit(dataset_root)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError("release dataset root must be an s3:// URI with a bucket")
    return parsed.netloc


def ec2_identity_document(*, timeout_seconds: float = 2.0) -> dict[str, str]:
    token_request = urllib.request.Request(
        f"{IMDS_BASE_URL}/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    )
    with urllib.request.urlopen(token_request, timeout=timeout_seconds) as response:
        token = response.read().decode("utf-8").strip()
    if not token:
        raise RuntimeError("EC2 IMDSv2 returned an empty token")
    document_request = urllib.request.Request(
        f"{IMDS_BASE_URL}/dynamic/instance-identity/document",
        headers={"X-aws-ec2-metadata-token": token},
    )
    with urllib.request.urlopen(document_request, timeout=timeout_seconds) as response:
        document = json.loads(response.read())
    if not isinstance(document, dict):
        raise ValueError("EC2 instance identity document must be an object")
    identity = {
        "instance_id": _non_empty_string(document.get("instanceId"), "instanceId"),
        "availability_zone": _non_empty_string(
            document.get("availabilityZone"), "availabilityZone"
        ),
        "compute_region": _non_empty_string(document.get("region"), "region"),
    }
    if not identity["availability_zone"].startswith(identity["compute_region"]):
        raise ValueError("EC2 availability zone does not belong to its declared region")
    return identity


def s3_bucket_region(bucket: str) -> str:
    result = subprocess.run(
        (
            "aws",
            "s3api",
            "get-bucket-location",
            "--bucket",
            bucket,
            "--query",
            "LocationConstraint",
            "--output",
            "text",
        ),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    location = result.stdout.strip()
    if location in {"", "None", "null"}:
        return "us-east-1"
    if location == "EU":
        return "eu-west-1"
    return location


def attest_same_region_s3(dataset_root: str) -> dict[str, Any]:
    bucket = s3_bucket(dataset_root)
    identity = ec2_identity_document()
    bucket_region = s3_bucket_region(bucket)
    if identity["compute_region"] != bucket_region:
        raise RuntimeError(
            f"EC2 region {identity['compute_region']} does not match "
            f"S3 bucket region {bucket_region}"
        )
    return {
        "schema_version": 1,
        "method": ATTESTATION_METHOD,
        **identity,
        "bucket": bucket,
        "bucket_region": bucket_region,
    }


def validate_same_region_s3_attestation(
    value: Any, dataset_root: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("release storage region attestation must be an object")
    missing = sorted(ATTESTATION_FIELDS - value.keys())
    unknown = sorted(value.keys() - ATTESTATION_FIELDS)
    if missing or unknown:
        raise ValueError(
            "release storage region attestation fields mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    if value["schema_version"] != 1 or value["method"] != ATTESTATION_METHOD:
        raise ValueError("release storage region attestation is unsupported")
    for field in ATTESTATION_FIELDS - {"schema_version"}:
        _non_empty_string(value[field], field)
    if value["bucket"] != s3_bucket(dataset_root):
        raise ValueError("release storage region attestation names the wrong bucket")
    if value["compute_region"] != value["bucket_region"]:
        raise ValueError("release host and S3 bucket regions differ")
    if not value["availability_zone"].startswith(value["compute_region"]):
        raise ValueError("release availability zone does not match compute region")
    return value
