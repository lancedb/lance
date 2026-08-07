#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION_SCRIPT="$SCRIPT_DIR/create-failure-issue.sh"

mock_gh() {
  local call_number=1
  local arg

  if [[ -f "$MOCK_GH_CALLS/count" ]]; then
    read -r call_number <"$MOCK_GH_CALLS/count"
    call_number=$((call_number + 1))
  fi
  printf '%s\n' "$call_number" >"$MOCK_GH_CALLS/count"
  for arg in "$@"; do
    printf '%s\0' "$arg"
  done >"$MOCK_GH_CALLS/$call_number"

  if [[ "${1:-}" == "issue" && "${2:-}" == "list" ]]; then
    printf '%s\n' "${MOCK_GH_LIST_RESPONSE:-[]}"
  elif [[ "${1:-}" == "issue" && "${2:-}" == "create" ]]; then
    printf '%s\n' "https://github.com/lance-format/lance/issues/999"
  fi
}

if [[ "$(basename "$0")" == "gh" ]]; then
  mock_gh "$@"
  exit 0
fi

TEST_TMP="$(mktemp -d)"
trap 'rm -rf "$TEST_TMP"' EXIT
mkdir -p "$TEST_TMP/bin" "$TEST_TMP/calls"
ln -s "$SCRIPT_DIR/test.sh" "$TEST_TMP/bin/gh"
MOCK_GH_CALLS="$TEST_TMP/calls"
export MOCK_GH_CALLS

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

assert_contains() {
  local haystack="$1"
  local needle="$2"
  [[ "$haystack" == *"$needle"* ]] || fail "expected '$needle' in '$haystack'"
}

assert_equals() {
  local expected="$1"
  local actual="$2"
  [[ "$expected" == "$actual" ]] || fail "expected '$expected', got '$actual'"
}

reset_mock() {
  rm -f "$MOCK_GH_CALLS"/count "$MOCK_GH_CALLS"/[0-9]*
  unset MOCK_GH_LIST_RESPONSE || true
}

run_action() {
  PATH="$TEST_TMP/bin:$PATH" \
    GH_TOKEN=test-token \
    JOB_RESULTS="$1" \
    WORKFLOW_NAME="Recurring Tests" \
    RUN_URL="https://github.com/lance-format/lance/actions/runs/123" \
    INCLUDE_CANCELLED="${2:-false}" \
    DEDUPLICATE="${3:-false}" \
    bash "$ACTION_SCRIPT"
}

load_call() {
  local call_number="$1"
  local arg
  CALL_ARGS=()
  while IFS= read -r -d '' arg; do
    CALL_ARGS+=("$arg")
  done <"$MOCK_GH_CALLS/$call_number"
}

test_failure_creates_issue() {
  local output

  reset_mock
  output="$(run_action '{"build":{"result":"failure"},"docs":{"result":"success"}}')"
  assert_contains "$output" "creating issue"
  assert_equals "1" "$(<"$MOCK_GH_CALLS/count")"
  load_call 1
  assert_equals "issue" "${CALL_ARGS[0]}"
  assert_equals "create" "${CALL_ARGS[1]}"
  assert_equals "Recurring Tests Failed (build)" "${CALL_ARGS[3]}"
  assert_contains "${CALL_ARGS[5]}" "**Failed jobs:** build"
  assert_contains "${CALL_ARGS[5]}" "actions/runs/123"
  assert_equals "ci" "${CALL_ARGS[7]}"
}

test_cancelled_is_opt_in() {
  local output

  reset_mock
  output="$(run_action '{"matrix":{"result":"cancelled"}}')"
  assert_contains "$output" "skipping issue creation"
  [[ ! -f "$MOCK_GH_CALLS/count" ]] || fail "cancelled job notified without opt-in"

  output="$(run_action '{"matrix":{"result":"cancelled"}}' true)"
  assert_equals "1" "$(<"$MOCK_GH_CALLS/count")"
  load_call 1
  assert_equals "Recurring Tests Cancelled (matrix)" "${CALL_ARGS[3]}"
  assert_contains "${CALL_ARGS[5]}" "**Cancelled jobs:** matrix"
}

test_deduplicate_seeds_new_issue() {
  reset_mock
  MOCK_GH_LIST_RESPONSE='[]'
  export MOCK_GH_LIST_RESPONSE

  run_action '{"build":{"result":"failure"}}' false true >/dev/null
  assert_equals "2" "$(<"$MOCK_GH_CALLS/count")"
  load_call 1
  assert_equals "list" "${CALL_ARGS[1]}"
  load_call 2
  assert_equals "create" "${CALL_ARGS[1]}"
  assert_contains "${CALL_ARGS[5]}" "<!-- create-failure-issue:Recurring%20Tests -->"
}

test_deduplicate_comments_on_open_issue() {
  reset_mock
  MOCK_GH_LIST_RESPONSE='[{"number":42,"body":"<!-- create-failure-issue:Recurring%20Tests -->\nPrevious run"}]'
  export MOCK_GH_LIST_RESPONSE

  run_action '{"build":{"result":"failure"}}' false true >/dev/null
  assert_equals "2" "$(<"$MOCK_GH_CALLS/count")"
  load_call 2
  assert_equals "issue" "${CALL_ARGS[0]}"
  assert_equals "comment" "${CALL_ARGS[1]}"
  assert_equals "42" "${CALL_ARGS[2]}"
  assert_equals "--body" "${CALL_ARGS[3]}"
  assert_contains "${CALL_ARGS[4]}" "**Failed jobs:** build"
  assert_contains "${CALL_ARGS[4]}" "actions/runs/123"
}

test_failure_creates_issue
test_cancelled_is_opt_in
test_deduplicate_seeds_new_issue
test_deduplicate_comments_on_open_issue

echo "All create-failure-issue tests passed"
