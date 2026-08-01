# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Black-box commit protocol conformance tests over S3-compatible services."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import urlsplit

import boto3
import pytest
from _commit_conformance import (
    CommitBackend,
    CommitOperation,
    DatasetHealth,
    DatasetState,
    OperationResult,
    assert_history_conforms,
    assert_results_match_visible_intents,
    assert_successful_result_is_visible,
    base_state,
    check_dataset_in_fresh_process,
    commit_request_occurrence,
    commit_request_operation,
    dataset_uri,
    execute_operation,
    finish_operation,
    foreign_writer_history_contract,
    operation_history_contract,
    recover_unknown_operation,
    start_operation,
    state_after_operation,
    stop_operation,
)
from _commit_fault_proxy import (
    CommitFaultProxy,
    FaultPlan,
    RequestTrace,
    read_fault_plans,
)
from botocore.config import Config

S3_ENDPOINT = os.environ.get("LANCE_CONFORMANCE_S3_ENDPOINT", "http://127.0.0.1:9000")
DYNAMODB_ENDPOINT = os.environ.get(
    "LANCE_CONFORMANCE_DYNAMODB_ENDPOINT", "http://127.0.0.1:4566"
)
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
LOCAL_AWS_ACCESS_KEY_ID = os.environ.get(
    "LANCE_CONFORMANCE_AWS_ACCESS_KEY_ID", "ACCESS_KEY"
)
LOCAL_AWS_SECRET_ACCESS_KEY = os.environ.get(
    "LANCE_CONFORMANCE_AWS_SECRET_ACCESS_KEY", "SECRET_KEY"
)

OPERATIONS: tuple[CommitOperation, ...] = (
    "append",
    "compaction",
    "add_columns",
    "create_index",
)
BACKENDS: tuple[CommitBackend, ...] = ("conditional", "dynamodb")


@dataclass(frozen=True)
class ConformanceCase:
    backend: CommitBackend
    operation: CommitOperation
    uri: str
    storage_options: dict[str, str]
    s3_proxy: CommitFaultProxy
    dynamodb_proxy: CommitFaultProxy

    @property
    def commit_proxy(self) -> CommitFaultProxy:
        return self.s3_proxy if self.backend == "conditional" else self.dynamodb_proxy


def boto3_client(service: str, *, endpoint_url: str | None = None):
    return boto3.client(
        service,
        endpoint_url=endpoint_url,
        region_name=AWS_REGION,
        aws_access_key_id=LOCAL_AWS_ACCESS_KEY_ID,
        aws_secret_access_key=LOCAL_AWS_SECRET_ACCESS_KEY,
    )


def delete_bucket(s3, bucket: str) -> None:
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
        if objects:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": objects})
    s3.delete_bucket(Bucket=bucket)


@pytest.fixture(scope="module")
def conformance_bucket() -> Iterator[str]:
    s3 = boto3_client("s3", endpoint_url=S3_ENDPOINT)
    bucket = f"lance-commit-conformance-{uuid.uuid4().hex[:16]}"
    s3.create_bucket(Bucket=bucket)
    yield bucket
    delete_bucket(s3, bucket)


@pytest.fixture(scope="module")
def conformance_dynamodb_table() -> Iterator[str]:
    dynamodb = boto3_client("dynamodb", endpoint_url=DYNAMODB_ENDPOINT)
    table = f"lance-commit-conformance-{uuid.uuid4().hex[:16]}"
    dynamodb.create_table(
        TableName=table,
        KeySchema=[
            {"AttributeName": "base_uri", "KeyType": "HASH"},
            {"AttributeName": "version", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "base_uri", "AttributeType": "S"},
            {"AttributeName": "version", "AttributeType": "N"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    dynamodb.get_waiter("table_exists").wait(TableName=table)
    yield table
    dynamodb.delete_table(TableName=table)


def storage_options(*, s3_endpoint: str, dynamodb_endpoint: str) -> dict[str, str]:
    return {
        "allow_http": "true",
        "aws_access_key_id": LOCAL_AWS_ACCESS_KEY_ID,
        "aws_secret_access_key": LOCAL_AWS_SECRET_ACCESS_KEY,
        "aws_region": AWS_REGION,
        "aws_endpoint": s3_endpoint,
        "aws_virtual_hosted_style_request": "false",
        "client_max_retries": "2",
        "client_retry_timeout": "5",
        "dynamodb_endpoint": dynamodb_endpoint,
    }


@contextmanager
def make_case(
    *,
    backend: CommitBackend,
    operation: CommitOperation,
    bucket: str,
    dynamodb_table: str,
) -> Iterator[ConformanceCase]:
    with (
        CommitFaultProxy(S3_ENDPOINT) as s3_proxy,
        CommitFaultProxy(DYNAMODB_ENDPOINT) as dynamodb_proxy,
    ):
        options = storage_options(
            s3_endpoint=s3_proxy.endpoint,
            dynamodb_endpoint=dynamodb_proxy.endpoint,
        )
        uri = dataset_uri(
            backend,
            bucket,
            f"{uuid.uuid4().hex}.lance",
            dynamodb_table=dynamodb_table,
        )
        setup = execute_operation(uri, options, "setup")
        assert setup.status == "success", setup
        s3_proxy.clear_traces()
        dynamodb_proxy.clear_traces()
        yield ConformanceCase(
            backend=backend,
            operation=operation,
            uri=uri,
            storage_options=options,
            s3_proxy=s3_proxy,
            dynamodb_proxy=dynamodb_proxy,
        )


def trace_directory(tmp_path: Path) -> Path:
    configured = os.environ.get("LANCE_COMMIT_CONFORMANCE_TRACE_DIR")
    return Path(configured) if configured else tmp_path / "commit-conformance-traces"


def persist_case_traces(
    case: ConformanceCase,
    tmp_path: Path,
    scenario: str,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    root = trace_directory(tmp_path)
    stem = f"{case.backend}-{case.operation}-{scenario}"
    shared_metadata = {
        "backend": case.backend,
        "operation": case.operation,
        "scenario": scenario,
        **(metadata or {}),
    }
    case.s3_proxy.write_trace(
        root / f"{stem}-s3.json",
        metadata={**shared_metadata, "service": "s3"},
    )
    case.dynamodb_proxy.write_trace(
        root / f"{stem}-dynamodb.json",
        metadata={**shared_metadata, "service": "dynamodb"},
    )


def assert_ambiguous_commit_trace(
    case: ConformanceCase, *, lose_verification_reads: bool
) -> None:
    operation = commit_request_operation(case.backend)
    traces = case.commit_proxy.traces()
    injected_commits = [
        trace
        for trace in traces
        if trace.operation == operation
        and trace.injected
        and trace.fault_phase == "after"
    ]
    assert len(injected_commits) == 1, traces
    commit = injected_commits[0]
    assert commit.forwarded
    assert commit.upstream_status is not None
    assert 200 <= commit.upstream_status < 300
    assert commit.client_status == 503
    assert any(
        trace.identity == commit.identity
        and trace.operation == operation
        and trace.upstream_status in {400, 409, 412}
        for trace in traces
    ), traces

    verification_operation = (
        "s3_manifest_read" if case.backend == "conditional" else "ddb_commit_read"
    )
    injected_verification = [
        trace
        for trace in traces
        if trace.identity == commit.identity
        and trace.operation == verification_operation
        and trace.injected
    ]
    assert bool(injected_verification) == lose_verification_reads


@pytest.mark.integration
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("lose_verification_reads", [False, True])
def test_commit_response_loss_preserves_final_state(
    conformance_bucket: str,
    conformance_dynamodb_table: str,
    tmp_path: Path,
    backend: CommitBackend,
    operation: CommitOperation,
    lose_verification_reads: bool,
) -> None:
    scenario = (
        "response-and-verification-lost" if lose_verification_reads else "response-lost"
    )
    with make_case(
        backend=backend,
        operation=operation,
        bucket=conformance_bucket,
        dynamodb_table=conformance_dynamodb_table,
    ) as case:
        plan = FaultPlan(
            operation=commit_request_operation(backend),
            occurrence=commit_request_occurrence(operation),
            verification_response_losses=100 if lose_verification_reads else 0,
        )
        case.commit_proxy.arm(plan)
        result: OperationResult | None = None
        try:
            result = execute_operation(case.uri, case.storage_options, operation)
            assert_ambiguous_commit_trace(
                case, lose_verification_reads=lose_verification_reads
            )
            case.s3_proxy.clear_faults()
            case.dynamodb_proxy.clear_faults()
            health = check_dataset_in_fresh_process(case.uri, case.storage_options)
            assert_history_conforms(
                health, operation_history_contract(operation, latest="after")
            )
            expected_status = "error" if lose_verification_reads else "success"
            assert result.status == expected_status, result
            assert_successful_result_is_visible(result, health, operation)
        finally:
            persist_case_traces(
                case,
                tmp_path,
                scenario,
                metadata={
                    "operation_result": asdict(result) if result is not None else None,
                },
            )


@pytest.mark.integration
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_foreign_writer_cannot_corrupt_visible_versions(
    conformance_bucket: str,
    conformance_dynamodb_table: str,
    tmp_path: Path,
    backend: CommitBackend,
    operation: CommitOperation,
) -> None:
    with make_case(
        backend=backend,
        operation=operation,
        bucket=conformance_bucket,
        dynamodb_table=conformance_dynamodb_table,
    ) as case:
        gate = tmp_path / f"start-{backend}-{operation}"
        case.commit_proxy.arm_barrier(
            operation=commit_request_operation(backend), parties=2
        )
        writer_a = start_operation(
            case.uri,
            case.storage_options,
            operation,
            "a",
            start_gate=gate,
        )
        writer_b = start_operation(
            case.uri,
            case.storage_options,
            operation,
            "b",
            start_gate=gate,
        )
        result_a: OperationResult | None = None
        result_b: OperationResult | None = None
        try:
            gate.touch()
            result_a = finish_operation(writer_a)
            result_b = finish_operation(writer_b)
            traces = case.commit_proxy.traces()
            operation_name = commit_request_operation(backend)
            commit_traces = [
                trace for trace in traces if trace.operation == operation_name
            ]
            assert any(
                trace.upstream_status is not None and 200 <= trace.upstream_status < 300
                for trace in commit_traces
            ), commit_traces
            assert any(
                trace.upstream_status in {400, 409, 412} for trace in commit_traces
            ), commit_traces
            case.s3_proxy.clear_faults()
            case.dynamodb_proxy.clear_faults()
            health = check_dataset_in_fresh_process(case.uri, case.storage_options)
            assert_history_conforms(health, foreign_writer_history_contract(operation))
            assert result_a.status in {"success", "error"}, result_a
            assert result_b.status in {"success", "error"}, result_b
            assert "success" in {result_a.status, result_b.status}
            assert_successful_result_is_visible(result_a, health, operation, "a")
            assert_successful_result_is_visible(result_b, health, operation, "b")
            assert_results_match_visible_intents(
                (result_a, result_b), health, operation
            )
        finally:
            stop_operation(writer_a)
            stop_operation(writer_b)
            persist_case_traces(
                case,
                tmp_path,
                "foreign-writer",
                metadata={
                    "writer_a": asdict(result_a) if result_a is not None else None,
                    "writer_b": asdict(result_b) if result_b is not None else None,
                },
            )


@pytest.mark.integration
@pytest.mark.recurring
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_fault_sweep_accepts_only_complete_states(
    conformance_bucket: str,
    conformance_dynamodb_table: str,
    tmp_path: Path,
    backend: CommitBackend,
    operation: CommitOperation,
) -> None:
    with make_case(
        backend=backend,
        operation=operation,
        bucket=conformance_bucket,
        dynamodb_table=conformance_dynamodb_table,
    ) as baseline:
        baseline_result = execute_operation(
            baseline.uri, baseline.storage_options, operation
        )
        assert baseline_result.status == "success", baseline_result
        mutation_counts = {
            "s3": sum(trace.operation != "" for trace in _mutations(baseline.s3_proxy)),
            "dynamodb": sum(
                trace.operation != "" for trace in _mutations(baseline.dynamodb_proxy)
            ),
        }

    for service, count in mutation_counts.items():
        for occurrence in range(1, count + 1):
            for phase in ("before", "after"):
                with make_case(
                    backend=backend,
                    operation=operation,
                    bucket=conformance_bucket,
                    dynamodb_table=conformance_dynamodb_table,
                ) as case:
                    proxy = case.s3_proxy if service == "s3" else case.dynamodb_proxy
                    plan = FaultPlan(
                        operation="mutation",
                        occurrence=occurrence,
                        phase=phase,
                        response_losses=100 if phase == "before" else 1,
                        verification_response_losses=100,
                    )
                    proxy.arm(plan)
                    scenario = f"sweep-{service}-{occurrence}-{phase}"
                    result: OperationResult | None = None
                    try:
                        result = execute_operation(
                            case.uri, case.storage_options, operation
                        )
                        assert any(trace.injected for trace in proxy.traces()), (
                            service,
                            occurrence,
                            phase,
                            proxy.traces(),
                        )
                        case.s3_proxy.clear_faults()
                        case.dynamodb_proxy.clear_faults()
                        health = check_dataset_in_fresh_process(
                            case.uri, case.storage_options
                        )
                        assert result.status in {"success", "error"}, result
                        assert_history_conforms(
                            health,
                            operation_history_contract(
                                operation,
                                latest=(
                                    "after" if result.status == "success" else "either"
                                ),
                            ),
                        )
                        assert_successful_result_is_visible(result, health, operation)
                    finally:
                        persist_case_traces(
                            case,
                            tmp_path,
                            scenario,
                            metadata={
                                "operation_result": (
                                    asdict(result) if result is not None else None
                                ),
                            },
                        )


def _mutations(proxy: CommitFaultProxy) -> list[RequestTrace]:
    return [
        trace
        for trace in proxy.traces()
        if trace.operation
        in {
            "s3_delete",
            "s3_copy",
            "s3_put",
            "s3_post",
            "s3_manifest_create",
            "ddb_commit_create",
            "ddb_commit_finalize",
            "ddb_put",
            "ddb_delete",
        }
    ]


@pytest.mark.integration
@pytest.mark.recurring
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("durable_commit", [False, True])
def test_unknown_outcome_recovery_applies_logical_operation_once(
    conformance_bucket: str,
    conformance_dynamodb_table: str,
    tmp_path: Path,
    backend: CommitBackend,
    operation: CommitOperation,
    durable_commit: bool,
) -> None:
    with make_case(
        backend=backend,
        operation=operation,
        bucket=conformance_bucket,
        dynamodb_table=conformance_dynamodb_table,
    ) as case:
        plan = FaultPlan(
            operation=commit_request_operation(backend),
            occurrence=commit_request_occurrence(operation),
            phase="after" if durable_commit else "before",
            response_losses=1 if durable_commit else 100,
            verification_response_losses=100 if durable_commit else 0,
        )
        case.commit_proxy.arm(plan)
        result: OperationResult | None = None
        try:
            result = execute_operation(case.uri, case.storage_options, operation)
            case.s3_proxy.clear_faults()
            case.dynamodb_proxy.clear_faults()
            recovery, health = recover_unknown_operation(
                case.uri, case.storage_options, operation
            )
            assert recovery == ("already_applied" if durable_commit else "retried")
            assert_history_conforms(
                health,
                operation_history_contract(
                    operation,
                    latest="after",
                    attempts=2 if recovery == "retried" else 1,
                ),
            )
            assert result.status == "error", result
        finally:
            persist_case_traces(
                case,
                tmp_path,
                f"recover-{'after' if durable_commit else 'before'}",
                metadata={
                    "operation_result": asdict(result) if result is not None else None,
                },
            )


@pytest.mark.integration
@pytest.mark.recurring
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_gc_after_unknown_outcome_preserves_committed_artifacts(
    conformance_bucket: str,
    conformance_dynamodb_table: str,
    tmp_path: Path,
    backend: CommitBackend,
    operation: CommitOperation,
) -> None:
    with make_case(
        backend=backend,
        operation=operation,
        bucket=conformance_bucket,
        dynamodb_table=conformance_dynamodb_table,
    ) as case:
        plan = FaultPlan(
            operation=commit_request_operation(backend),
            occurrence=commit_request_occurrence(operation),
            verification_response_losses=100,
        )
        case.commit_proxy.arm(plan)
        result: OperationResult | None = None
        try:
            result = execute_operation(case.uri, case.storage_options, operation)
            case.s3_proxy.clear_faults()
            case.dynamodb_proxy.clear_faults()
            cleanup = execute_operation(case.uri, case.storage_options, "cleanup")
            assert cleanup.status == "success", cleanup
            health = check_dataset_in_fresh_process(case.uri, case.storage_options)
            assert_history_conforms(
                health, operation_history_contract(operation, latest="after")
            )
            assert result.status == "error", result
        finally:
            persist_case_traces(
                case,
                tmp_path,
                "gc-after-unknown",
                metadata={
                    "operation_result": asdict(result) if result is not None else None,
                },
            )


@pytest.mark.integration
@pytest.mark.recurring
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_process_exit_after_durable_commit_preserves_state(
    conformance_bucket: str,
    conformance_dynamodb_table: str,
    tmp_path: Path,
    backend: CommitBackend,
    operation: CommitOperation,
) -> None:
    with make_case(
        backend=backend,
        operation=operation,
        bucket=conformance_bucket,
        dynamodb_table=conformance_dynamodb_table,
    ) as case:
        gate = tmp_path / f"kill-start-{backend}-{operation}"
        process = start_operation(
            case.uri,
            case.storage_options,
            operation,
            start_gate=gate,
        )
        plan = FaultPlan(
            operation=commit_request_operation(backend),
            occurrence=commit_request_occurrence(operation),
            kill_pid=process.pid,
        )
        case.commit_proxy.arm(plan)
        result: OperationResult | None = None
        try:
            gate.touch()
            result = finish_operation(process)
            case.s3_proxy.clear_faults()
            case.dynamodb_proxy.clear_faults()
            health = check_dataset_in_fresh_process(case.uri, case.storage_options)
            assert_history_conforms(
                health, operation_history_contract(operation, latest="after")
            )
            assert result.status == "killed", result
        finally:
            stop_operation(process)
            persist_case_traces(
                case,
                tmp_path,
                "process-exit",
                metadata={
                    "operation_result": asdict(result) if result is not None else None,
                },
            )


@pytest.mark.integration
@pytest.mark.recurring
def test_fault_plan_trace_is_replayable(
    conformance_bucket: str,
    conformance_dynamodb_table: str,
    tmp_path: Path,
) -> None:
    original = FaultPlan(operation="s3_manifest_create", occurrence=1)
    plans = [original]
    normalized_traces: list[list[tuple[object, ...]]] = []
    for run in (1, 2):
        plan = plans[-1]
        with make_case(
            backend="conditional",
            operation="append",
            bucket=conformance_bucket,
            dynamodb_table=conformance_dynamodb_table,
        ) as case:
            case.s3_proxy.arm(plan)
            try:
                result = execute_operation(case.uri, case.storage_options, "append")
                assert result.status == "success", result
                operation_traces = case.s3_proxy.traces()
                case.s3_proxy.clear_faults()
                health = check_dataset_in_fresh_process(case.uri, case.storage_options)
                assert_history_conforms(
                    health, operation_history_contract("append", latest="after")
                )
                assert_successful_result_is_visible(result, health, "append")
                normalized_traces.append(
                    [
                        (
                            trace.operation,
                            trace.forwarded,
                            trace.upstream_status,
                            trace.client_status,
                            trace.injected,
                            trace.fault_phase,
                        )
                        for trace in operation_traces
                    ]
                )
            finally:
                persist_case_traces(case, tmp_path, f"replay-{run}")
        if run == 1:
            artifact = trace_directory(tmp_path) / "conditional-append-replay-1-s3.json"
            replayed = read_fault_plans(artifact)
            assert replayed == [original]
            plans.append(replayed[0])
    assert normalized_traces[0] == normalized_traces[1]


@pytest.mark.integration
@pytest.mark.recurring
def test_fresh_process_oracle_detects_missing_index_artifact(
    conformance_bucket: str,
    conformance_dynamodb_table: str,
    tmp_path: Path,
) -> None:
    with make_case(
        backend="conditional",
        operation="create_index",
        bucket=conformance_bucket,
        dynamodb_table=conformance_dynamodb_table,
    ) as case:
        result = execute_operation(case.uri, case.storage_options, "create_index")
        assert result.status == "success", result
        check_dataset_in_fresh_process(case.uri, case.storage_options)

        parsed = urlsplit(case.uri)
        prefix = parsed.path.lstrip("/")
        s3 = boto3_client("s3", endpoint_url=S3_ENDPOINT)
        objects = s3.list_objects_v2(Bucket=parsed.netloc, Prefix=prefix).get(
            "Contents", []
        )
        index_artifacts = [
            item["Key"] for item in objects if "/_indices/" in item["Key"]
        ]
        assert index_artifacts, objects
        s3.delete_object(Bucket=parsed.netloc, Key=index_artifacts[0])

        with pytest.raises(AssertionError, match="fresh-process dataset validation"):
            check_dataset_in_fresh_process(case.uri, case.storage_options)
        persist_case_traces(case, tmp_path, "negative-missing-index-artifact")


@pytest.mark.integration
@pytest.mark.recurring
@pytest.mark.real_s3
@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("lose_verification_reads", [False, True])
def test_real_s3_commit_response_loss(
    tmp_path: Path,
    operation: CommitOperation,
    lose_verification_reads: bool,
) -> None:
    bucket = os.environ.get("LANCE_CONFORMANCE_REAL_S3_BUCKET")
    if bucket is None:
        pytest.skip("LANCE_CONFORMANCE_REAL_S3_BUCKET is not configured")
    if os.environ.get("LANCE_CONFORMANCE_REAL_S3_ISOLATED") != "1":
        pytest.skip("real-S3 faults must be run through the fail-closed CI runner")

    region = os.environ.get("LANCE_CONFORMANCE_REAL_S3_REGION", "us-east-2")
    listen_port = int(os.environ.get("LANCE_CONFORMANCE_PROXY_PORT", "18080"))
    upstream_connect_host = os.environ["LANCE_CONFORMANCE_REAL_S3_UPSTREAM_IP"]
    upstream = f"https://s3.{region}.amazonaws.com"
    client_endpoint = f"http://s3.{region}.amazonaws.com"
    uri = f"s3://{bucket}/commit-conformance/{uuid.uuid4().hex}.lance"
    options = {
        "allow_http": "true",
        "aws_region": region,
        "aws_endpoint": client_endpoint,
        "aws_virtual_hosted_style_request": "false",
        "client_max_retries": "2",
        "client_retry_timeout": "5",
    }

    parsed = urlsplit(uri)
    s3 = boto3.client(
        "s3",
        region_name=region,
        endpoint_url=client_endpoint,
        config=Config(s3={"addressing_style": "path"}),
    )
    with CommitFaultProxy(
        upstream,
        listen_port=listen_port,
        upstream_connect_host=upstream_connect_host,
    ) as proxy:
        plan: FaultPlan | None = None
        result: OperationResult | None = None
        try:
            setup = execute_operation(uri, options, "setup")
            assert setup.status == "success", (
                "real-S3 transparent proxy is not reachable; configure the host's "
                f"port-80 redirect to 127.0.0.1:{listen_port}",
                setup,
            )
            assert any(
                trace.operation == "s3_manifest_create" for trace in proxy.traces()
            ), (
                "real-S3 setup bypassed the transparent fault proxy; the test "
                "cannot prove that response loss was injected",
                proxy.traces(),
            )
            proxy.clear_traces()
            plan = FaultPlan(
                operation="s3_manifest_create",
                occurrence=commit_request_occurrence(operation),
                verification_response_losses=100 if lose_verification_reads else 0,
            )
            proxy.arm(plan)
            result = execute_operation(uri, options, operation)
            traces = proxy.traces()
            assert any(
                trace.operation == "s3_manifest_create"
                and trace.injected
                and trace.fault_phase == "after"
                for trace in traces
            ), (
                "the real-S3 commit response-loss fault was not exercised",
                traces,
            )
            assert any(trace.operation == "s3_manifest_read" for trace in traces), (
                "the conditional commit did not perform read-back after the "
                "lost response",
                traces,
            )
            if lose_verification_reads:
                assert any(
                    trace.operation == "s3_manifest_read"
                    and trace.injected
                    and trace.fault_phase == "verification"
                    for trace in traces
                ), (
                    "the real-S3 verification response-loss fault was not exercised",
                    traces,
                )
            proxy.clear_faults()
            health = check_dataset_in_fresh_process(uri, options)
            assert_history_conforms(
                health, operation_history_contract(operation, latest="after")
            )
            expected_status = "error" if lose_verification_reads else "success"
            assert result.status == expected_status, result
            assert_successful_result_is_visible(result, health, operation)
        finally:
            proxy.clear_faults()
            if plan is not None:
                proxy.write_trace(
                    trace_directory(tmp_path)
                    / f"real-s3-{operation}-{lose_verification_reads}.json",
                    metadata={
                        "backend": "real-s3",
                        "operation": operation,
                        "lose_verification_reads": lose_verification_reads,
                        "operation_result": (
                            asdict(result) if result is not None else None
                        ),
                    },
                )
            objects = s3.list_objects_v2(
                Bucket=bucket, Prefix=parsed.path.lstrip("/")
            ).get("Contents", [])
            if objects:
                s3.delete_objects(
                    Bucket=bucket,
                    Delete={"Objects": [{"Key": item["Key"]} for item in objects]},
                )


def test_final_state_oracle_rejects_partial_and_duplicate_states() -> None:
    before = base_state()
    after = state_after_operation("append")

    def health_for(states: list[DatasetState]) -> DatasetHealth:
        versions = list(range(1, len(states) + 1))
        return {
            "versions": versions,
            "opened_version": versions[-1],
            "states": {str(version): state for version, state in zip(versions, states)},
            "transactions": {
                str(version): {
                    "uuid": f"transaction-{version}",
                    "operation": "TestOperation",
                }
                for version in versions
            },
        }

    contract = operation_history_contract("append", latest="after")
    assert_history_conforms(health_for([before, after]), contract)

    duplicate = json.loads(json.dumps(after))
    duplicate["rows"].append({"id": 200})
    with pytest.raises(AssertionError):
        assert_history_conforms(health_for([before, duplicate]), contract)

    partial = json.loads(json.dumps(after))
    partial["rows"] = partial["rows"][:-1]
    with pytest.raises(AssertionError):
        assert_history_conforms(health_for([before, partial]), contract)

    with pytest.raises(AssertionError, match="too many versions"):
        assert_history_conforms(health_for([before, after, after]), contract)

    with pytest.raises(AssertionError):
        assert_history_conforms(health_for([before, after, before]), contract)

    stale_open = health_for([before, after])
    stale_open["opened_version"] = 1
    with pytest.raises(AssertionError, match="latest visible version"):
        assert_history_conforms(stale_open, contract)

    duplicate_transaction = health_for([before, after])
    duplicate_transaction["transactions"]["2"]["uuid"] = "transaction-1"
    with pytest.raises(AssertionError, match="committed more than once"):
        assert_history_conforms(duplicate_transaction, contract)

    compaction_retry = health_for(
        [before, before, before, state_after_operation("compaction")]
    )
    assert_history_conforms(
        compaction_retry,
        operation_history_contract("compaction", latest="after", attempts=2),
    )
    with pytest.raises(AssertionError, match="too many versions"):
        assert_history_conforms(
            compaction_retry,
            operation_history_contract("compaction", latest="after"),
        )

    compacted = state_after_operation("compaction")
    concurrent_compaction = health_for([before, before, compacted, compacted])
    concurrent_compaction["transactions"]["1"]["operation"] = "Overwrite"
    concurrent_compaction["transactions"]["2"]["operation"] = "BaseOperation"
    concurrent_compaction["transactions"]["3"]["operation"] = "Rewrite"
    concurrent_compaction["transactions"]["4"]["operation"] = "BaseOperation"
    assert_history_conforms(
        concurrent_compaction, foreign_writer_history_contract("compaction")
    )
    compaction_result = OperationResult(
        status="success",
        returncode=0,
        error_type=None,
        error_message=None,
        committed_version=3,
        transaction_uuid="transaction-3",
        stdout="",
        stderr="",
    )
    assert_results_match_visible_intents(
        (compaction_result,), concurrent_compaction, "compaction"
    )

    duplicate_rewrite = json.loads(json.dumps(concurrent_compaction))
    duplicate_rewrite["transactions"]["4"]["operation"] = "Rewrite"
    with pytest.raises(AssertionError, match="visible commit intents"):
        assert_results_match_visible_intents(
            (compaction_result,), duplicate_rewrite, "compaction"
        )


def test_local_oracle_correlates_success_with_visible_transaction(
    tmp_path: Path,
) -> None:
    uri = str(tmp_path / "local-oracle.lance")
    setup = execute_operation(uri, {}, "setup")
    assert setup.status == "success", setup
    result = execute_operation(uri, {}, "append")
    assert result.status == "success", result

    health = check_dataset_in_fresh_process(uri, {})
    assert_history_conforms(
        health, operation_history_contract("append", latest="after")
    )
    assert_successful_result_is_visible(result, health, "append")


def test_operation_timeout_reaps_child() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    with pytest.raises(AssertionError, match="timed out"):
        finish_operation(process, timeout=0.01)
    assert process.poll() is not None


def test_trace_rebinds_process_kill_fault(tmp_path: Path) -> None:
    trace = tmp_path / "kill-fault.json"
    with CommitFaultProxy("http://127.0.0.1:9") as proxy:
        proxy.arm(FaultPlan(operation="mutation", kill_pid=12345))
        proxy.write_trace(trace)

    assert '"kill_pid"' not in trace.read_text()
    with pytest.raises(ValueError, match="requires a new kill_pid"):
        read_fault_plans(trace)
    replayed = read_fault_plans(trace, kill_pid=54321)
    assert replayed[0].kill_pid == 54321
