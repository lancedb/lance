#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

set -euo pipefail

if [[ -z "${LANCE_CONFORMANCE_REAL_S3_BUCKET:-}" ]]; then
  echo "LANCE_CONFORMANCE_REAL_S3_BUCKET must name an existing test bucket" >&2
  exit 2
fi

for command in flock getent iptables setsid sudo; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "${command} is required by the fail-closed real-S3 runner" >&2
    exit 2
  fi
done
sudo -n true

region="${LANCE_CONFORMANCE_REAL_S3_REGION:-us-east-2}"
proxy_port="${LANCE_CONFORMANCE_PROXY_PORT:-18080}"
s3_host="s3.${region}.amazonaws.com"
test_user_id="$(id -u)"
nat_chain="LANCE_CC_S3"
guard_chain="LANCE_CC_S3_GUARD"
hosts_marker="# lance-commit-conformance"
lock_file="/tmp/lance-commit-conformance.lock"
test_pid=""

exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "another real-S3 commit conformance run holds ${lock_file}" >&2
  exit 2
fi

remove_stale_network_state() {
  sudo iptables -t nat -D OUTPUT \
    -p tcp -d 127.0.0.1/32 --dport 80 \
    -m owner --uid-owner "${test_user_id}" -j "${nat_chain}" \
    2>/dev/null || true
  sudo iptables -D OUTPUT \
    -p tcp -d 127.0.0.1/32 --dport 80 \
    -m owner --uid-owner "${test_user_id}" -j "${guard_chain}" \
    2>/dev/null || true
  sudo iptables -t nat -F "${nat_chain}" 2>/dev/null || true
  sudo iptables -t nat -X "${nat_chain}" 2>/dev/null || true
  sudo iptables -F "${guard_chain}" 2>/dev/null || true
  sudo iptables -X "${guard_chain}" 2>/dev/null || true
  sudo sed -i "\|${hosts_marker}$|d" /etc/hosts
}

cleanup() {
  status=$?
  trap - EXIT INT TERM
  set +e
  if [[ -n "${test_pid}" ]] && kill -0 "${test_pid}" 2>/dev/null; then
    kill -TERM -- "-${test_pid}" 2>/dev/null || true
    for _ in {1..20}; do
      kill -0 "${test_pid}" 2>/dev/null || break
      sleep 0.1
    done
    kill -KILL -- "-${test_pid}" 2>/dev/null || true
    wait "${test_pid}" 2>/dev/null || true
  fi
  remove_stale_network_state
  if grep -Fq "${hosts_marker}" /etc/hosts; then
    echo "failed to remove the real-S3 /etc/hosts isolation entry" >&2
    status=1
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Recover exact state left by a previously killed runner before resolving the
# real upstream address. The generated /etc/hosts entry maps only the regional
# S3 hostname and is tagged with hosts_marker.
remove_stale_network_state
upstream_ip="$(getent ahostsv4 "${s3_host}" | awk 'NR == 1 { print $1 }')"
if [[ -z "${upstream_ip}" || "${upstream_ip}" == "127.0.0.1" ]]; then
  echo "could not resolve a non-loopback IPv4 address for ${s3_host}" >&2
  exit 2
fi

# Resolve the test client's S3 hostname to loopback. Its only port-80 route is
# redirected to the fault proxy. The filter guard rejects the request if the
# NAT redirect disappears, so signed plaintext cannot escape to real S3.
printf '127.0.0.1 %s %s\n' "${s3_host}" "${hosts_marker}" | sudo tee -a /etc/hosts >/dev/null
sudo iptables -t nat -N "${nat_chain}"
sudo iptables -t nat -A "${nat_chain}" -j REDIRECT --to-ports "${proxy_port}"
sudo iptables -t nat -I OUTPUT 1 \
  -p tcp -d 127.0.0.1/32 --dport 80 \
  -m owner --uid-owner "${test_user_id}" -j "${nat_chain}"
sudo iptables -N "${guard_chain}"
sudo iptables -A "${guard_chain}" -j REJECT
sudo iptables -I OUTPUT 1 \
  -p tcp -d 127.0.0.1/32 --dport 80 \
  -m owner --uid-owner "${test_user_id}" -j "${guard_chain}"

cd "$(dirname "$0")/../python"
export LANCE_COMMIT_CONFORMANCE_TRACE_DIR="${LANCE_COMMIT_CONFORMANCE_TRACE_DIR:-target/commit-conformance-traces}"
export LANCE_CONFORMANCE_REAL_S3_ISOLATED=1
export LANCE_CONFORMANCE_REAL_S3_UPSTREAM_IP="${upstream_ip}"
setsid uv run --frozen pytest --run-integration -m "recurring and real_s3" -q \
  python/tests/test_commit_conformance.py &
test_pid=$!
wait "${test_pid}"
test_pid=""
