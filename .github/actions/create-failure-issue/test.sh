#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION_SCRIPT="$SCRIPT_DIR/create-failure-issue.sh"

acquire_mock_state_lock() {
  local attempt

  for ((attempt = 0; attempt < 1000; attempt++)); do
    mkdir "$MOCK_GH_STATE_DIR/lock" 2>/dev/null && return
    sleep 0.01
  done

  echo "Timed out waiting for mock state lock" >&2
  exit 1
}

release_mock_state_lock() {
  rmdir "$MOCK_GH_STATE_DIR/lock"
}

mock_stateful_gh() {
  local args=("$@")
  local body=""
  local close_comment=""
  local initial_list_call="false"
  local initial_list_count
  local issue_number
  local next_issue_number
  local attempt
  local index
  local tmp_file

  if [[ "${1:-}" == "issue" && "${2:-}" == "list" ]]; then
    # Hold the first two lookups at a barrier so both actions observe no issue.
    acquire_mock_state_lock
    read -r initial_list_count <"$MOCK_GH_STATE_DIR/initial-list-count"
    if (( initial_list_count < 2 )); then
      initial_list_count=$((initial_list_count + 1))
      printf '%s\n' "$initial_list_count" >"$MOCK_GH_STATE_DIR/initial-list-count"
      initial_list_call="true"
    fi
    release_mock_state_lock

    if [[ "$initial_list_call" == "true" ]]; then
      for ((attempt = 0; attempt < 1000; attempt++)); do
        acquire_mock_state_lock
        read -r initial_list_count <"$MOCK_GH_STATE_DIR/initial-list-count"
        release_mock_state_lock
        (( initial_list_count >= 2 )) && break
        sleep 0.01
      done
      if (( initial_list_count < 2 )); then
        echo "Timed out waiting for both initial issue-list calls" >&2
        exit 1
      fi
      printf '%s\n' '[]'
      return
    fi

    acquire_mock_state_lock
    jq '[.[] | select(.state == "open") | {number, body}]' \
      "$MOCK_GH_STATE_DIR/issues.json"
    release_mock_state_lock
    return
  fi

  for ((index = 0; index < ${#args[@]}; index++)); do
    if [[ "${args[index]}" == "--body" ]]; then
      body="${args[index + 1]}"
    elif [[ "${args[index]}" == "--comment" ]]; then
      close_comment="${args[index + 1]}"
    fi
  done

  if [[ "${1:-}" == "issue" && "${2:-}" == "create" ]]; then
    acquire_mock_state_lock
    read -r next_issue_number <"$MOCK_GH_STATE_DIR/next-issue-number"
    issue_number="$next_issue_number"
    printf '%s\n' "$((next_issue_number + 1))" >"$MOCK_GH_STATE_DIR/next-issue-number"
    tmp_file="$MOCK_GH_STATE_DIR/issues.json.$$"
    jq \
      --argjson number "$issue_number" \
      --arg body "$body" \
      '. + [{number: $number, body: $body, state: "open"}]' \
      "$MOCK_GH_STATE_DIR/issues.json" >"$tmp_file"
    mv "$tmp_file" "$MOCK_GH_STATE_DIR/issues.json"
    release_mock_state_lock
    printf 'https://github.com/lance-format/lance/issues/%s\n' "$issue_number"
  elif [[ "${1:-}" == "issue" && "${2:-}" == "comment" ]]; then
    issue_number="${3:?issue number is required}"
    acquire_mock_state_lock
    tmp_file="$MOCK_GH_STATE_DIR/comments.json.$$"
    jq \
      --argjson number "$issue_number" \
      --arg body "$body" \
      '. + [{number: $number, body: $body}]' \
      "$MOCK_GH_STATE_DIR/comments.json" >"$tmp_file"
    mv "$tmp_file" "$MOCK_GH_STATE_DIR/comments.json"
    release_mock_state_lock
  elif [[ "${1:-}" == "issue" && "${2:-}" == "close" ]]; then
    issue_number="${3:?issue number is required}"
    acquire_mock_state_lock
    tmp_file="$MOCK_GH_STATE_DIR/issues.json.$$"
    jq \
      --argjson number "$issue_number" \
      'map(if .number == $number then .state = "closed" else . end)' \
      "$MOCK_GH_STATE_DIR/issues.json" >"$tmp_file"
    mv "$tmp_file" "$MOCK_GH_STATE_DIR/issues.json"
    tmp_file="$MOCK_GH_STATE_DIR/closes.json.$$"
    jq \
      --argjson number "$issue_number" \
      --arg comment "$close_comment" \
      '. + [{number: $number, comment: $comment}]' \
      "$MOCK_GH_STATE_DIR/closes.json" >"$tmp_file"
    mv "$tmp_file" "$MOCK_GH_STATE_DIR/closes.json"
    release_mock_state_lock
  fi
}

mock_gh() {
  local call_number=1
  local arg

  if [[ -n "${MOCK_GH_STATE_DIR:-}" ]]; then
    mock_stateful_gh "$@"
    return
  fi

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
  unset MOCK_GH_STATE_DIR || true
}

run_action() {
  PATH="$TEST_TMP/bin:$PATH" \
    GH_TOKEN=test-token \
    JOB_RESULTS="$1" \
    WORKFLOW_NAME="Recurring Tests" \
    RUN_URL="${4:-https://github.com/lance-format/lance/actions/runs/123}" \
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
  assert_equals "3" "$(<"$MOCK_GH_CALLS/count")"
  load_call 1
  assert_equals "list" "${CALL_ARGS[1]}"
  load_call 2
  assert_equals "create" "${CALL_ARGS[1]}"
  assert_contains "${CALL_ARGS[5]}" "<!-- create-failure-issue:Recurring%20Tests -->"
  load_call 3
  assert_equals "list" "${CALL_ARGS[1]}"
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

test_deduplicate_reconciles_concurrent_creates() {
  local canonical_issue_number
  local closed_issue_number
  local combined_notifications
  local first_pid
  local open_issue_body
  local second_pid
  local state_dir="$TEST_TMP/concurrent-state"

  reset_mock
  rm -rf "$state_dir"
  mkdir -p "$state_dir"
  printf '%s\n' '0' >"$state_dir/initial-list-count"
  printf '%s\n' '1000' >"$state_dir/next-issue-number"
  printf '%s\n' '[]' >"$state_dir/issues.json"
  printf '%s\n' '[]' >"$state_dir/comments.json"
  printf '%s\n' '[]' >"$state_dir/closes.json"
  MOCK_GH_STATE_DIR="$state_dir"
  export MOCK_GH_STATE_DIR

  run_action \
    '{"build":{"result":"failure"}}' \
    false \
    true \
    'https://github.com/lance-format/lance/actions/runs/100' \
    >"$state_dir/first-output" &
  first_pid=$!
  run_action \
    '{"build":{"result":"failure"}}' \
    false \
    true \
    'https://github.com/lance-format/lance/actions/runs/200' \
    >"$state_dir/second-output" &
  second_pid=$!

  wait "$first_pid" || fail "first concurrent action failed"
  wait "$second_pid" || fail "second concurrent action failed"
  unset MOCK_GH_STATE_DIR

  assert_equals "1" "$(jq '[.[] | select(.state == "open")] | length' "$state_dir/issues.json")"
  assert_equals "1" "$(jq '[.[] | select(.state == "closed")] | length' "$state_dir/issues.json")"
  assert_equals "1" "$(jq 'length' "$state_dir/comments.json")"
  assert_equals "1" "$(jq 'length' "$state_dir/closes.json")"

  canonical_issue_number="$(
    jq -r '.[] | select(.state == "open") | .number' "$state_dir/issues.json"
  )"
  closed_issue_number="$(
    jq -r '.[] | select(.state == "closed") | .number' "$state_dir/issues.json"
  )"
  assert_equals "$canonical_issue_number" "$(jq -r '.[0].number' "$state_dir/comments.json")"
  assert_equals "$closed_issue_number" "$(jq -r '.[0].number' "$state_dir/closes.json")"
  assert_contains "$(jq -r '.[0].comment' "$state_dir/closes.json")" "#$canonical_issue_number"

  open_issue_body="$(
    jq -r '.[] | select(.state == "open") | .body' "$state_dir/issues.json"
  )"
  combined_notifications="$open_issue_body
$(jq -r '.[0].body' "$state_dir/comments.json")"
  assert_contains "$combined_notifications" "actions/runs/100"
  assert_contains "$combined_notifications" "actions/runs/200"
}

test_failure_creates_issue
test_cancelled_is_opt_in
test_deduplicate_seeds_new_issue
test_deduplicate_comments_on_open_issue
test_deduplicate_reconciles_concurrent_creates

echo "All create-failure-issue tests passed"
