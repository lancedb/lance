# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Deterministic storage fault injection for commit conformance tests."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import socket
import ssl
import threading
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Literal, cast
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from email.message import Message
    from pathlib import Path


@dataclass(frozen=True)
class RequestTrace:
    """One request as observed on both sides of the proxy."""

    sequence: int
    method: str
    path: str
    operation: str
    identity: str
    resource: str
    request_sha256: str
    forwarded: bool
    upstream_status: int | None
    client_status: int | None
    injected: bool
    fault_phase: str | None


@dataclass(frozen=True)
class FaultPlan:
    """A serializable deterministic fault at a classified request boundary."""

    operation: str
    occurrence: int = 1
    phase: Literal["before", "after"] = "after"
    response_losses: int = 1
    verification_response_losses: int = 0
    kill_pid: int | None = None

    def __post_init__(self) -> None:
        if self.occurrence < 1:
            raise ValueError(f"occurrence must be at least 1, got {self.occurrence}")
        if self.response_losses < 1:
            raise ValueError(
                f"response_losses must be at least 1, got {self.response_losses}"
            )
        if self.verification_response_losses < 0:
            raise ValueError(
                "verification_response_losses must be non-negative, got "
                f"{self.verification_response_losses}"
            )


@dataclass(frozen=True)
class _ClassifiedRequest:
    method: str
    path: str
    operation: str
    identity: str
    resource: str
    request_sha256: str
    mutation: bool


@dataclass
class _ArmedFault:
    plan: FaultPlan
    matches: int = 0
    response_losses_remaining: int = 0
    target_identity: str | None = None
    target_resource: str | None = None

    def __post_init__(self) -> None:
        self.response_losses_remaining = self.plan.response_losses


@dataclass
class _VerificationFault:
    operation: str
    identity: str
    remaining: int


@dataclass
class _BarrierFault:
    operation: str
    parties: int
    barrier: threading.Barrier
    identity: str | None = None
    arrivals: int = 0


@dataclass(frozen=True)
class _ResponseAction:
    kind: Literal["none", "replace", "kill"]
    phase: str | None = None
    pid: int | None = None


class _ProxyState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._fault: _ArmedFault | None = None
        self._verification_fault: _VerificationFault | None = None
        self._barrier_fault: _BarrierFault | None = None
        self._traces: list[RequestTrace] = []
        self._next_sequence = 1

    def arm(self, plan: FaultPlan) -> None:
        with self._lock:
            self._fault = _ArmedFault(plan)
            self._verification_fault = None

    def arm_barrier(self, *, operation: str, parties: int) -> None:
        if parties < 2:
            raise ValueError(f"parties must be at least 2, got {parties}")
        with self._lock:
            self._barrier_fault = _BarrierFault(
                operation=operation,
                parties=parties,
                barrier=threading.Barrier(parties, timeout=30),
            )

    def clear_faults(self) -> None:
        with self._lock:
            self._fault = None
            self._verification_fault = None
            self._barrier_fault = None

    def clear_traces(self) -> None:
        with self._lock:
            self._traces.clear()
            self._next_sequence = 1

    def wait_at_barrier(self, request: _ClassifiedRequest) -> None:
        barrier: threading.Barrier | None = None
        with self._lock:
            fault = self._barrier_fault
            if fault is None or not self._matches_operation(fault.operation, request):
                return
            if fault.identity is None:
                fault.identity = request.identity
            if request.identity != fault.identity or fault.arrivals >= fault.parties:
                return
            fault.arrivals += 1
            barrier = fault.barrier
        assert barrier is not None
        barrier.wait()

    def before_request(self, request: _ClassifiedRequest) -> _ResponseAction:
        with self._lock:
            fault = self._fault
            if fault is None or fault.plan.phase != "before":
                return _ResponseAction("none")
            if not self._fault_targets_request(fault, request):
                return _ResponseAction("none")
            fault.response_losses_remaining -= 1
            if fault.response_losses_remaining == 0:
                self._fault = None
            return _ResponseAction("replace", "before")

    def after_response(
        self, request: _ClassifiedRequest, upstream_status: int
    ) -> _ResponseAction:
        if not 200 <= upstream_status < 300:
            return _ResponseAction("none")

        with self._lock:
            verification = self._verification_fault
            if (
                verification is not None
                and request.operation == verification.operation
                and request.identity == verification.identity
            ):
                verification.remaining -= 1
                if verification.remaining == 0:
                    self._verification_fault = None
                return _ResponseAction("replace", "verification")

            fault = self._fault
            if fault is None or fault.plan.phase != "after":
                return _ResponseAction("none")
            if not self._fault_targets_request(fault, request):
                return _ResponseAction("none")

            if fault.plan.verification_response_losses > 0:
                verification_operation = {
                    "s3_manifest_create": "s3_manifest_read",
                    "ddb_commit_create": "ddb_commit_read",
                }.get(request.operation)
                if verification_operation is not None:
                    self._verification_fault = _VerificationFault(
                        operation=verification_operation,
                        identity=request.identity,
                        remaining=fault.plan.verification_response_losses,
                    )

            fault.response_losses_remaining -= 1
            if fault.response_losses_remaining == 0:
                self._fault = None
            if fault.plan.kill_pid is not None:
                return _ResponseAction("kill", "after", fault.plan.kill_pid)
            return _ResponseAction("replace", "after")

    def _fault_targets_request(
        self, fault: _ArmedFault, request: _ClassifiedRequest
    ) -> bool:
        if fault.target_identity is not None:
            return (
                request.identity == fault.target_identity
                and request.resource == fault.target_resource
            )
        if not self._matches_operation(fault.plan.operation, request):
            return False
        fault.matches += 1
        if fault.matches != fault.plan.occurrence:
            return False
        fault.target_identity = request.identity
        fault.target_resource = request.resource
        return True

    @staticmethod
    def _matches_operation(operation: str, request: _ClassifiedRequest) -> bool:
        return operation == request.operation or (
            operation == "mutation" and request.mutation
        )

    def record(
        self,
        request: _ClassifiedRequest,
        *,
        forwarded: bool,
        upstream_status: int | None,
        client_status: int | None,
        injected: bool,
        fault_phase: str | None,
    ) -> None:
        with self._lock:
            sequence = self._next_sequence
            self._next_sequence += 1
            self._traces.append(
                RequestTrace(
                    sequence=sequence,
                    method=request.method,
                    path=request.path,
                    operation=request.operation,
                    identity=request.identity,
                    resource=request.resource,
                    request_sha256=request.request_sha256,
                    forwarded=forwarded,
                    upstream_status=upstream_status,
                    client_status=client_status,
                    injected=injected,
                    fault_phase=fault_phase,
                )
            )

    def traces(self) -> list[RequestTrace]:
        with self._lock:
            return list(self._traces)


class _ProxyServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        upstream_scheme: str,
        upstream_host: str,
        upstream_port: int,
        upstream_connect_host: str | None,
        state: _ProxyState,
    ) -> None:
        super().__init__(server_address, _CommitProxyHandler)
        self.upstream_scheme = upstream_scheme
        self.upstream_host = upstream_host
        self.upstream_port = upstream_port
        self.upstream_connect_host = upstream_connect_host
        self.state = state


class _FixedAddressHTTPSConnection(http.client.HTTPSConnection):
    """Connect to a fixed address while authenticating the original TLS host."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_host: str,
        timeout: float,
    ) -> None:
        self._tls_context = ssl.create_default_context()
        super().__init__(host, port, timeout=timeout, context=self._tls_context)
        self._fixed_connect_host = connect_host

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._fixed_connect_host, self.port),
            self.timeout,
        )
        self.sock = self._tls_context.wrap_socket(self.sock, server_hostname=self.host)


class _CommitProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_DELETE(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_GET(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_HEAD(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_PATCH(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_POST(self) -> None:  # noqa: N802
        self._proxy_request()

    def do_PUT(self) -> None:  # noqa: N802
        self._proxy_request()

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _proxy_request(self) -> None:
        server = cast("_ProxyServer", self.server)
        body = self._read_request_body()
        request = _classify_request(self.command, self.path, self.headers, body)

        before_action = server.state.before_request(request)
        if before_action.kind == "replace":
            self._send_injected_failure(request)
            server.state.record(
                request,
                forwarded=False,
                upstream_status=None,
                client_status=503,
                injected=True,
                fault_phase=before_action.phase,
            )
            return

        server.state.wait_at_barrier(request)
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower()
            not in {"connection", "content-length", "expect", "transfer-encoding"}
        }
        headers["Content-Length"] = str(len(body))

        if server.upstream_scheme == "https":
            if server.upstream_connect_host is None:
                upstream: http.client.HTTPConnection = http.client.HTTPSConnection(
                    server.upstream_host,
                    server.upstream_port,
                    timeout=30,
                )
            else:
                upstream = _FixedAddressHTTPSConnection(
                    server.upstream_host,
                    server.upstream_port,
                    connect_host=server.upstream_connect_host,
                    timeout=30,
                )
        else:
            upstream = http.client.HTTPConnection(
                server.upstream_connect_host or server.upstream_host,
                server.upstream_port,
                timeout=30,
            )
        try:
            upstream.request(self.command, self.path, body=body, headers=headers)
            response = upstream.getresponse()
            response_body = response.read()
            upstream_status = response.status
            upstream_headers = response.getheaders()
        finally:
            upstream.close()

        action = server.state.after_response(request, upstream_status)
        if action.kind == "kill":
            server.state.record(
                request,
                forwarded=True,
                upstream_status=upstream_status,
                client_status=None,
                injected=True,
                fault_phase=action.phase,
            )
            self.close_connection = True
            assert action.pid is not None
            os.kill(action.pid, 9)
            return

        if action.kind == "replace":
            client_status = 503
            self._send_injected_failure(request)
        else:
            client_status = upstream_status
            self._send_upstream_response(
                status=upstream_status,
                headers=upstream_headers,
                body=response_body,
            )

        server.state.record(
            request,
            forwarded=True,
            upstream_status=upstream_status,
            client_status=client_status,
            injected=action.kind != "none",
            fault_phase=action.phase,
        )

    def _read_request_body(self) -> bytes:
        content_length = self.headers.get("Content-Length")
        if content_length is not None:
            return self.rfile.read(int(content_length))
        if self.headers.get("Transfer-Encoding", "").lower() != "chunked":
            return b""

        body = bytearray()
        while True:
            size_line = self.rfile.readline()
            if not size_line:
                raise ConnectionError("unexpected EOF while reading chunk size")
            size = int(size_line.split(b";", 1)[0].strip(), 16)
            if size == 0:
                while self.rfile.readline() not in {b"\r\n", b"\n", b""}:
                    pass
                break
            body.extend(self.rfile.read(size))
            if self.rfile.read(2) != b"\r\n":
                raise ConnectionError("invalid chunk terminator")
        return bytes(body)

    def _send_injected_failure(self, request: _ClassifiedRequest) -> None:
        if request.operation.startswith("ddb_"):
            body = json.dumps(
                {
                    "__type": "com.amazonaws.dynamodb.v20120810#InternalServerError",
                    "message": "injected commit conformance failure",
                }
            ).encode()
            content_type = "application/x-amz-json-1.0"
        else:
            body = (
                b'<?xml version="1.0" encoding="UTF-8"?>'
                b"<Error><Code>InternalError</Code>"
                b"<Message>injected commit conformance failure</Message>"
                b"<RequestId>lance-commit-conformance</RequestId></Error>"
            )
            content_type = "application/xml"
        self.send_response(503)
        self.send_header("Content-Type", content_type)
        self.send_header(
            "Content-Length", "0" if self.command == "HEAD" else str(len(body))
        )
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        self.close_connection = True

    def _send_upstream_response(
        self,
        *,
        status: int,
        headers: list[tuple[str, str]],
        body: bytes,
    ) -> None:
        self.send_response(status)
        for key, value in headers:
            if key.lower() not in {
                "connection",
                "content-length",
                "date",
                "server",
                "transfer-encoding",
            }:
                self.send_header(key, value)
        if self.command == "HEAD":
            content_length = next(
                (value for key, value in headers if key.lower() == "content-length"),
                "0",
            )
            self.send_header("Content-Length", content_length)
        else:
            self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        self.close_connection = True


def _classify_request(
    method: str,
    path: str,
    headers: Message[str, str],
    body: bytes,
) -> _ClassifiedRequest:
    target = headers.get("X-Amz-Target", "")
    if target:
        return _classify_dynamodb_request(method, path, target, body)
    return _classify_s3_request(method, path, headers, body)


def _classify_dynamodb_request(
    method: str, path: str, target: str, body: bytes
) -> _ClassifiedRequest:
    operation_name = target.rsplit(".", 1)[-1]
    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError:
        payload = {}
    item = payload.get("Item") or payload.get("Key") or {}
    base_uri = item.get("base_uri", {}).get("S", "")
    version = item.get("version", {}).get("N", "")
    identity = f"{base_uri}@{version}" if base_uri or version else target
    resource = item.get("path", {}).get("S", identity)
    condition = payload.get("ConditionExpression", "")

    if operation_name == "PutItem" and "attribute_not_exists" in condition:
        operation = "ddb_commit_create"
    elif operation_name == "PutItem" and "attribute_exists" in condition:
        operation = "ddb_commit_finalize"
    elif operation_name == "PutItem":
        operation = "ddb_put"
    elif operation_name == "GetItem":
        operation = "ddb_commit_read"
    elif operation_name == "Query":
        operation = "ddb_query"
    elif operation_name == "DeleteItem":
        operation = "ddb_delete"
    else:
        operation = f"ddb_{operation_name.lower()}"

    return _ClassifiedRequest(
        method=method,
        path=path,
        operation=operation,
        identity=identity,
        resource=resource,
        request_sha256=hashlib.sha256(body).hexdigest(),
        mutation=operation
        in {
            "ddb_commit_create",
            "ddb_commit_finalize",
            "ddb_put",
            "ddb_delete",
        },
    )


def _classify_s3_request(
    method: str,
    path: str,
    headers: Message[str, str],
    body: bytes,
) -> _ClassifiedRequest:
    request_path = urlsplit(path).path
    is_manifest = "/_versions/" in request_path and ".manifest" in request_path
    if (
        method == "PUT"
        and headers.get("If-None-Match") == "*"
        and is_manifest
        and request_path.endswith(".manifest")
    ):
        operation = "s3_manifest_create"
    elif method in {"GET", "HEAD"} and is_manifest:
        operation = "s3_manifest_read"
    elif method == "DELETE":
        operation = "s3_delete"
    elif method == "PUT" and headers.get("X-Amz-Copy-Source"):
        operation = "s3_copy"
    elif method == "PUT":
        operation = "s3_put"
    elif method == "POST":
        operation = "s3_post"
    else:
        operation = f"s3_{method.lower()}"

    return _ClassifiedRequest(
        method=method,
        path=path,
        operation=operation,
        identity=request_path,
        resource=request_path,
        request_sha256=hashlib.sha256(body).hexdigest(),
        mutation=method in {"DELETE", "PATCH", "POST", "PUT"},
    )


class CommitFaultProxy:
    """Forward storage traffic while injecting replayable response faults."""

    def __init__(
        self,
        upstream_endpoint: str,
        *,
        listen_port: int = 0,
        upstream_connect_host: str | None = None,
    ) -> None:
        upstream = urlsplit(upstream_endpoint)
        if upstream.scheme not in {"http", "https"} or upstream.hostname is None:
            raise ValueError(
                f"upstream_endpoint must be an HTTP(S) URL, got {upstream_endpoint}"
            )
        self._state = _ProxyState()
        self._server = _ProxyServer(
            ("127.0.0.1", listen_port),
            upstream.scheme,
            upstream.hostname,
            upstream.port or (443 if upstream.scheme == "https" else 80),
            upstream_connect_host,
            self._state,
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._fault_plans: list[FaultPlan] = []

    @property
    def endpoint(self) -> str:
        host, port = cast("tuple[str, int]", self._server.server_address)
        return f"http://{host}:{port}"

    @property
    def port(self) -> int:
        _, port = cast("tuple[str, int]", self._server.server_address)
        return port

    def arm(self, plan: FaultPlan) -> None:
        self._fault_plans.append(plan)
        self._state.arm(plan)

    def arm_barrier(self, *, operation: str, parties: int = 2) -> None:
        self._state.arm_barrier(operation=operation, parties=parties)

    def clear_faults(self) -> None:
        self._state.clear_faults()

    def clear_traces(self) -> None:
        self._state.clear_traces()
        self._fault_plans.clear()

    def traces(self) -> list[RequestTrace]:
        return self._state.traces()

    def write_trace(
        self,
        path: Path,
        *,
        metadata: dict[str, object] | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": metadata or {},
            "fault_plans": [
                {
                    **{
                        key: value
                        for key, value in asdict(plan).items()
                        if key != "kill_pid"
                    },
                    "kill_process": plan.kill_pid is not None,
                }
                for plan in self._fault_plans
            ],
            "requests": [asdict(trace) for trace in self.traces()],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def __enter__(self) -> CommitFaultProxy:
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def read_fault_plans(path: Path, *, kill_pid: int | None = None) -> list[FaultPlan]:
    """Load fault plans, rebinding any process-kill action explicitly."""

    payload = json.loads(path.read_text())
    if any(plan.get("kill_process") for plan in payload.get("fault_plans", [])):
        if kill_pid is None:
            raise ValueError("replaying a process-kill fault requires a new kill_pid")
    return [
        FaultPlan(
            operation=str(raw["operation"]),
            occurrence=int(raw["occurrence"]),
            phase=cast("Literal['before', 'after']", raw["phase"]),
            response_losses=int(raw["response_losses"]),
            verification_response_losses=int(raw["verification_response_losses"]),
            kill_pid=kill_pid if raw.get("kill_process") else None,
        )
        for raw in payload.get("fault_plans", [])
    ]
