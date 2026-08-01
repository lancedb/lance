# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Public-API drivers and final-state checks for commit conformance tests."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Mapping, TypedDict, cast

if TYPE_CHECKING:
    from pathlib import Path

CommitBackend = Literal["conditional", "dynamodb"]
CommitOperation = Literal["append", "compaction", "add_columns", "create_index"]
MaintenanceOperation = Literal["setup", "cleanup"]
WriterVariant = Literal["main", "a", "b"]

BASE_ROW_COUNT = 200
APPEND_ROWS = {
    "main": list(range(200, 220)),
    "a": list(range(200, 220)),
    "b": list(range(300, 320)),
}
ADDED_COLUMNS = {
    "main": ("double_id", "id * 2"),
    "a": ("writer_a", "id + 1000"),
    "b": ("writer_b", "id + 2000"),
}
INDEX_NAMES = {
    "main": "id_idx",
    "a": "id_idx_a",
    "b": "id_idx_b",
}


@dataclass(frozen=True)
class BackendContract:
    uri_scheme: str
    commit_request_operation: str
    requires_dynamodb_table: bool = False


@dataclass(frozen=True)
class OperationContract:
    commit_request_occurrence: int = 1
    max_before_versions: int = 1


BACKEND_CONTRACTS: Mapping[CommitBackend, BackendContract] = {
    "conditional": BackendContract(
        uri_scheme="s3", commit_request_operation="s3_manifest_create"
    ),
    "dynamodb": BackendContract(
        uri_scheme="s3+ddb",
        commit_request_operation="ddb_commit_create",
        requires_dynamodb_table=True,
    ),
}

OPERATION_CONTRACTS: Mapping[CommitOperation, OperationContract] = {
    "append": OperationContract(),
    "compaction": OperationContract(commit_request_occurrence=2, max_before_versions=2),
    "add_columns": OperationContract(),
    "create_index": OperationContract(),
}


class DatasetState(TypedDict):
    rows: list[dict[str, int]]
    schema: list[tuple[str, str]]
    fragments: int
    indices: list[dict[str, object]]
    index_queries: dict[str, list[int]]


class TransactionState(TypedDict):
    uuid: str
    operation: str


class DatasetHealth(TypedDict):
    versions: list[int]
    opened_version: int
    states: dict[str, DatasetState]
    transactions: dict[str, TransactionState]


@dataclass(frozen=True)
class OperationResult:
    status: Literal["success", "error", "killed", "crashed"]
    returncode: int
    error_type: str | None
    error_message: str | None
    committed_version: int | None
    transaction_uuid: str | None
    stdout: str
    stderr: str


_OPERATION_SCRIPT = r"""
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import lance
import pyarrow as pa

uri = os.environ["LANCE_CONFORMANCE_URI"]
storage_options = json.loads(os.environ["LANCE_CONFORMANCE_STORAGE_OPTIONS"])
operation = os.environ["LANCE_CONFORMANCE_OPERATION"]
variant = os.environ.get("LANCE_CONFORMANCE_VARIANT", "main")
gate = os.environ.get("LANCE_CONFORMANCE_START_GATE")
if gate:
    deadline = time.monotonic() + 30
    while not Path(gate).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"start gate was not released: {gate}")
        time.sleep(0.01)

try:
    result_dataset = None
    if operation == "setup":
        result_dataset = lance.write_dataset(
            pa.table({"id": pa.array(range(200), type=pa.int64())}),
            uri,
            max_rows_per_file=100,
            storage_options=storage_options,
        )
    elif operation == "append":
        ranges = {
            "main": range(200, 220),
            "a": range(200, 220),
            "b": range(300, 320),
        }
        result_dataset = lance.write_dataset(
            pa.table({"id": pa.array(ranges[variant], type=pa.int64())}),
            uri,
            mode="append",
            storage_options=storage_options,
        )
    elif operation == "cleanup":
        dataset = lance.dataset(uri, storage_options=storage_options)
        dataset.cleanup_old_versions(
            older_than=timedelta(0),
            retain_versions=100,
            delete_unverified=True,
        )
    else:
        dataset = lance.dataset(uri, storage_options=storage_options)
        if operation == "compaction":
            dataset.optimize.compact_files(
                target_rows_per_fragment=1000,
                materialize_deletions=False,
                num_threads=1,
            )
        elif operation == "add_columns":
            columns = {
                "main": ("double_id", "id * 2"),
                "a": ("writer_a", "id + 1000"),
                "b": ("writer_b", "id + 2000"),
            }
            name, expression = columns[variant]
            dataset.add_columns({name: expression})
        elif operation == "create_index":
            names = {"main": "id_idx", "a": "id_idx_a", "b": "id_idx_b"}
            dataset.create_scalar_index("id", "BTREE", name=names[variant])
        else:
            raise ValueError(f"unknown operation: {operation}")
        result_dataset = dataset

    result = {"status": "success"}
    if result_dataset is not None:
        committed_version = result_dataset.version
        transaction = result_dataset.read_transaction(committed_version)
        if transaction is None:
            raise AssertionError(
                "successful operation has no transaction at version "
                f"{committed_version}"
            )
        result.update({
            "committed_version": committed_version,
            "transaction_uuid": transaction.uuid,
        })
    print(json.dumps(result))
except Exception as error:
    print(json.dumps({
        "status": "error",
        "error_type": type(error).__name__,
        "error_message": str(error),
    }))
"""


_HEALTH_SCRIPT = r"""
import json
import os

import lance

uri = os.environ["LANCE_CONFORMANCE_URI"]
storage_options = json.loads(os.environ["LANCE_CONFORMANCE_STORAGE_OPTIONS"])
dataset = lance.dataset(uri, storage_options=storage_options)
versions = sorted(item["version"] for item in dataset.versions())
states = {}
transactions = {}
for version in versions:
    snapshot = dataset.checkout_version(version)
    snapshot.validate()
    transaction = snapshot.read_transaction(version)
    if transaction is None:
        raise AssertionError(f"version {version} has no readable transaction")
    transactions[str(version)] = {
        "uuid": transaction.uuid,
        "operation": type(transaction.operation).__name__,
    }
    table = snapshot.to_table()
    rows = sorted(table.to_pylist(), key=lambda row: row["id"])
    indices = []
    index_queries = {}
    btree_id_indices = []
    for index in snapshot.list_indices():
        normalized = {
            "name": index["name"],
            "type": index["type"],
            "fields": sorted(index["fields"]),
        }
        indices.append(normalized)
        index_stats = snapshot.stats.index_stats(index["name"])
        if not index_stats:
            raise AssertionError(f"index {index['name']} has no readable statistics")
        if index["type"] == "BTree" and index["fields"] == ["id"]:
            btree_id_indices.append(index["name"])
    if btree_id_indices:
        scanner = snapshot.scanner(filter="id >= 190", use_scalar_index=True)
        plan = scanner.explain_plan()
        if "ScalarIndexQuery" not in plan or not any(
            name in plan for name in btree_id_indices
        ):
            raise AssertionError(
                f"no BTree id index was used by the health query: {plan}"
            )
        query_rows = sorted(scanner.to_table()["id"].to_pylist())
        index_queries.update({name: query_rows for name in btree_id_indices})
    states[str(version)] = {
        "rows": rows,
        "schema": [[field.name, str(field.type)] for field in table.schema],
        "fragments": len(snapshot.get_fragments()),
        "indices": sorted(indices, key=lambda index: index["name"]),
        "index_queries": index_queries,
    }

print(json.dumps({
    "versions": versions,
    "opened_version": dataset.version,
    "states": states,
    "transactions": transactions,
}, sort_keys=True))
"""


def dataset_uri(
    backend: CommitBackend,
    bucket: str,
    key: str,
    *,
    dynamodb_table: str | None,
) -> str:
    contract = BACKEND_CONTRACTS[backend]
    if contract.requires_dynamodb_table and dynamodb_table is None:
        raise ValueError("dynamodb_table is required for the dynamodb backend")
    query = (
        f"?ddbTableName={dynamodb_table}" if contract.requires_dynamodb_table else ""
    )
    return f"{contract.uri_scheme}://{bucket}/{key}{query}"


def commit_request_operation(backend: CommitBackend) -> str:
    return BACKEND_CONTRACTS[backend].commit_request_operation


def commit_request_occurrence(operation: CommitOperation) -> int:
    return OPERATION_CONTRACTS[operation].commit_request_occurrence


def operation_environment(
    uri: str,
    storage_options: dict[str, str],
    operation: CommitOperation | MaintenanceOperation,
    variant: WriterVariant = "main",
    *,
    start_gate: Path | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env["LANCE_CONFORMANCE_URI"] = uri
    env["LANCE_CONFORMANCE_STORAGE_OPTIONS"] = json.dumps(storage_options)
    env["LANCE_CONFORMANCE_OPERATION"] = operation
    env["LANCE_CONFORMANCE_VARIANT"] = variant
    if start_gate is not None:
        env["LANCE_CONFORMANCE_START_GATE"] = str(start_gate)
    return env


def start_operation(
    uri: str,
    storage_options: dict[str, str],
    operation: CommitOperation | MaintenanceOperation,
    variant: WriterVariant = "main",
    *,
    start_gate: Path | None = None,
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", _OPERATION_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=operation_environment(
            uri,
            storage_options,
            operation,
            variant,
            start_gate=start_gate,
        ),
        text=True,
    )


def finish_operation(
    process: subprocess.Popen[str], *, timeout: float = 180
) -> OperationResult:
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        process.kill()
        stdout, stderr = process.communicate()
        raise AssertionError(
            f"operation timed out after {timeout} seconds and was reaped\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        ) from error

    if process.returncode != 0:
        returncode = cast("int", process.returncode)
        return OperationResult(
            status="killed" if returncode == -signal.SIGKILL else "crashed",
            returncode=returncode,
            error_type=None,
            error_message=None,
            committed_version=None,
            transaction_uuid=None,
            stdout=stdout,
            stderr=stderr,
        )
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError(f"operation produced no result\nstderr:\n{stderr}")
    payload = json.loads(lines[-1])
    return OperationResult(
        status=payload["status"],
        returncode=0,
        error_type=payload.get("error_type"),
        error_message=payload.get("error_message"),
        committed_version=payload.get("committed_version"),
        transaction_uuid=payload.get("transaction_uuid"),
        stdout=stdout,
        stderr=stderr,
    )


def stop_operation(process: subprocess.Popen[str]) -> None:
    """Kill and reap a child that a failed assertion would otherwise leak."""

    if process.poll() is None:
        process.kill()
    process.communicate()


def execute_operation(
    uri: str,
    storage_options: dict[str, str],
    operation: CommitOperation | MaintenanceOperation,
    variant: WriterVariant = "main",
    *,
    timeout: float = 180,
) -> OperationResult:
    return finish_operation(
        start_operation(uri, storage_options, operation, variant), timeout=timeout
    )


def check_dataset_in_fresh_process(
    uri: str, storage_options: dict[str, str], *, timeout: float = 180
) -> DatasetHealth:
    env = os.environ.copy()
    env["LANCE_CONFORMANCE_URI"] = uri
    env["LANCE_CONFORMANCE_STORAGE_OPTIONS"] = json.dumps(storage_options)
    result = subprocess.run(
        [sys.executable, "-c", _HEALTH_SCRIPT],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, (
        "fresh-process dataset validation failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines, f"fresh-process health check produced no output: {result.stderr}"
    return cast("DatasetHealth", json.loads(lines[-1]))


def base_state() -> DatasetState:
    return {
        "rows": [{"id": value} for value in range(BASE_ROW_COUNT)],
        "schema": [("id", "int64")],
        "fragments": 2,
        "indices": [],
        "index_queries": {},
    }


def state_after_operation(
    operation: CommitOperation, variant: WriterVariant = "main"
) -> DatasetState:
    state = base_state()
    if operation == "append":
        state["rows"] = state["rows"] + [
            {"id": value} for value in APPEND_ROWS[variant]
        ]
        state["fragments"] = 3
    elif operation == "compaction":
        state["fragments"] = 1
    elif operation == "add_columns":
        name, _ = ADDED_COLUMNS[variant]
        offset = {"main": 0, "a": 1000, "b": 2000}[variant]
        multiplier = 2 if variant == "main" else 1
        state["rows"] = [
            {"id": value, name: value * multiplier + offset}
            for value in range(BASE_ROW_COUNT)
        ]
        state["schema"] = [("id", "int64"), (name, "int64")]
    elif operation == "create_index":
        name = INDEX_NAMES[variant]
        state["indices"] = [{"name": name, "type": "BTree", "fields": ["id"]}]
        state["index_queries"] = {name: list(range(190, BASE_ROW_COUNT))}
    return state


def combined_foreign_writer_states(operation: CommitOperation) -> list[DatasetState]:
    states = [base_state(), state_after_operation(operation, "a")]
    state_b = state_after_operation(operation, "b")
    if state_b not in states:
        states.append(state_b)

    if operation == "append":
        both = base_state()
        both["rows"] = both["rows"] + [
            {"id": value} for value in APPEND_ROWS["a"] + APPEND_ROWS["b"]
        ]
        both["fragments"] = 4
        states.append(both)
    elif operation == "add_columns":
        both = base_state()
        both["rows"] = [
            {"id": value, "writer_a": value + 1000, "writer_b": value + 2000}
            for value in range(BASE_ROW_COUNT)
        ]
        both["schema"] = [
            ("id", "int64"),
            ("writer_a", "int64"),
            ("writer_b", "int64"),
        ]
        states.append(both)
        reversed_schema = cast("DatasetState", json.loads(json.dumps(both)))
        reversed_schema["schema"] = [
            ("id", "int64"),
            ("writer_b", "int64"),
            ("writer_a", "int64"),
        ]
        states.append(reversed_schema)
    elif operation == "create_index":
        both = base_state()
        both["indices"] = [
            {"name": INDEX_NAMES["a"], "type": "BTree", "fields": ["id"]},
            {"name": INDEX_NAMES["b"], "type": "BTree", "fields": ["id"]},
        ]
        both["index_queries"] = {
            INDEX_NAMES["a"]: list(range(190, BASE_ROW_COUNT)),
            INDEX_NAMES["b"]: list(range(190, BASE_ROW_COUNT)),
        }
        states.append(both)
    return states


def state_signature(state: DatasetState) -> str:
    return json.dumps(state, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class HistoryNode:
    name: str
    state: DatasetState
    max_versions: int = 1


@dataclass(frozen=True)
class HistoryContract:
    nodes: tuple[HistoryNode, ...]
    transitions: frozenset[tuple[str, str]]
    latest_nodes: frozenset[str]


def operation_history_contract(
    operation: CommitOperation,
    *,
    latest: Literal["before", "after", "either"],
    attempts: int = 1,
) -> HistoryContract:
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")
    operation_contract = OPERATION_CONTRACTS[operation]
    latest_nodes = {
        "before": frozenset({"before"}),
        "after": frozenset({"after"}),
        "either": frozenset({"before", "after"}),
    }[latest]
    return HistoryContract(
        nodes=(
            HistoryNode(
                "before",
                base_state(),
                max_versions=(
                    1 + attempts * (operation_contract.max_before_versions - 1)
                ),
            ),
            HistoryNode("after", state_after_operation(operation)),
        ),
        transitions=frozenset({("before", "after")}),
        latest_nodes=latest_nodes,
    )


def foreign_writer_history_contract(operation: CommitOperation) -> HistoryContract:
    states = combined_foreign_writer_states(operation)
    if operation == "compaction":
        # Each writer may reserve fragment IDs before one rewrite wins.
        return HistoryContract(
            nodes=(
                HistoryNode("before", states[0], max_versions=3),
                HistoryNode("after", states[1]),
            ),
            transitions=frozenset({("before", "after")}),
            latest_nodes=frozenset({"after"}),
        )

    nodes = [HistoryNode("before", states[0])]
    nodes.extend(
        HistoryNode(f"effect_{index}", state)
        for index, state in enumerate(states[1:], start=1)
    )
    combined_start = 3
    transitions = {
        ("before", "effect_1"),
        ("before", "effect_2"),
    }
    transitions.update(
        (single, f"effect_{index}")
        for single in ("effect_1", "effect_2")
        for index in range(combined_start, len(states))
    )
    return HistoryContract(
        nodes=tuple(nodes),
        transitions=frozenset(transitions),
        latest_nodes=frozenset(node.name for node in nodes[1:]),
    )


def recover_unknown_operation(
    uri: str,
    storage_options: dict[str, str],
    operation: CommitOperation,
) -> tuple[Literal["already_applied", "retried"], DatasetHealth]:
    before = base_state()
    after = state_after_operation(operation)
    health = check_dataset_in_fresh_process(uri, storage_options)
    latest = health["states"][str(health["versions"][-1])]
    if state_signature(latest) == state_signature(after):
        return "already_applied", health
    if state_signature(latest) != state_signature(before):
        raise AssertionError(
            f"cannot recover operation from a partial or foreign state: {latest}"
        )

    retry = execute_operation(uri, storage_options, operation)
    assert retry.status == "success", retry
    return "retried", check_dataset_in_fresh_process(uri, storage_options)


def assert_history_conforms(
    health: DatasetHealth,
    contract: HistoryContract,
) -> None:
    versions = health["versions"]
    assert versions, "dataset has no visible versions"
    assert versions == list(range(1, versions[-1] + 1)), (
        f"visible versions are not contiguous: {versions}"
    )
    assert health["opened_version"] == versions[-1], (
        "opening the dataset did not resolve to the latest visible version: "
        f"opened={health['opened_version']}, visible={versions}"
    )

    transaction_versions = {int(version) for version in health["transactions"]}
    assert transaction_versions == set(versions), (
        "transaction metadata does not cover every visible version: "
        f"transactions={sorted(transaction_versions)}, visible={versions}"
    )
    transaction_uuids = [
        health["transactions"][str(version)]["uuid"] for version in versions
    ]
    assert all(transaction_uuids), "every visible version must have a transaction UUID"
    assert len(transaction_uuids) == len(set(transaction_uuids)), (
        f"a transaction UUID was committed more than once: {transaction_uuids}"
    )

    nodes_by_signature = {state_signature(node.state): node for node in contract.nodes}
    assert len(nodes_by_signature) == len(contract.nodes), (
        "history contract contains indistinguishable state nodes"
    )
    observed_nodes: list[str] = []
    node_counts: dict[str, int] = {}
    for version in versions:
        state = health["states"][str(version)]
        signature = state_signature(state)
        node = nodes_by_signature.get(signature)
        assert node is not None, (
            f"version {version} is not a complete contract state: {state}"
        )
        observed_nodes.append(node.name)
        node_counts[node.name] = node_counts.get(node.name, 0) + 1
        assert node_counts[node.name] <= node.max_versions, (
            f"state {node.name!r} appears in too many versions: {observed_nodes}"
        )
        ids = [row["id"] for row in state["rows"]]
        assert len(ids) == len(set(ids)), (
            f"version {version} contains duplicate row IDs"
        )

    assert observed_nodes[0] == "before", (
        f"history must start from the setup state: {observed_nodes}"
    )
    for previous, current in zip(observed_nodes, observed_nodes[1:]):
        if previous != current:
            assert (previous, current) in contract.transitions, (
                f"illegal history transition {previous!r} -> {current!r}: "
                f"{observed_nodes}"
            )
    assert observed_nodes[-1] in contract.latest_nodes, (
        f"latest state {observed_nodes[-1]!r} is not allowed: {observed_nodes}"
    )


def assert_successful_result_is_visible(
    result: OperationResult,
    health: DatasetHealth,
    operation: CommitOperation,
    variant: WriterVariant = "main",
) -> None:
    if result.status != "success":
        return
    assert result.committed_version is not None, result
    assert result.transaction_uuid is not None, result
    assert result.committed_version in health["versions"], (
        f"successful operation returned invisible version {result.committed_version}: "
        f"{health['versions']}"
    )
    transaction = health["transactions"][str(result.committed_version)]
    assert transaction["uuid"] == result.transaction_uuid, (
        "successful operation does not own its reported visible version: "
        f"result={result}, transaction={transaction}"
    )
    expected_transaction_operation = {
        "append": "Append",
        "compaction": "Rewrite",
        "add_columns": "Project",
        "create_index": "CreateIndex",
    }[operation]
    assert transaction["operation"] == expected_transaction_operation, (
        "successful operation reported a version committed by a different intent: "
        f"expected={expected_transaction_operation}, transaction={transaction}"
    )

    state = health["states"][str(result.committed_version)]
    if operation == "append":
        visible_ids = {row["id"] for row in state["rows"]}
        assert set(APPEND_ROWS[variant]).issubset(visible_ids), (
            f"successful append effect is absent at version {result.committed_version}"
        )
    elif operation == "compaction":
        assert state["fragments"] == 1, (
            "successful compaction effect is absent at version "
            f"{result.committed_version}"
        )
    elif operation == "add_columns":
        column, _ = ADDED_COLUMNS[variant]
        assert column in {name for name, _ in state["schema"]}, (
            "successful add-columns effect is absent at version "
            f"{result.committed_version}"
        )
    elif operation == "create_index":
        index_name = INDEX_NAMES[variant]
        assert index_name in {index["name"] for index in state["indices"]}, (
            "successful create-index effect is absent at version "
            f"{result.committed_version}"
        )
