#!/usr/bin/env bash

set -euo pipefail

: "${JOB_RESULTS:?JOB_RESULTS must be set}"
: "${WORKFLOW_NAME:?WORKFLOW_NAME must be set}"
: "${RUN_URL:?RUN_URL must be set}"

INCLUDE_CANCELLED="${INCLUDE_CANCELLED:-false}"
DEDUPLICATE="${DEDUPLICATE:-false}"

validate_boolean() {
  local name="$1"
  local value="$2"

  if [[ "$value" != "true" && "$value" != "false" ]]; then
    echo "$name must be 'true' or 'false', got '$value'" >&2
    exit 2
  fi
}

validate_boolean "INCLUDE_CANCELLED" "$INCLUDE_CANCELLED"
validate_boolean "DEDUPLICATE" "$DEDUPLICATE"

find_matching_issues() {
  local marker="$1"

  gh issue list \
    --state open \
    --label ci \
    --limit 1000 \
    --json number,body |
    jq -c --arg marker "$marker" '
      map(select((.body // "") | contains($marker)))
      | sort_by(.number)
    '
}

load_issue_contents() {
  local issue_number="$1"

  gh issue view "$issue_number" --json body,comments |
    jq -r '[.body // "", (.comments[]?.body // "")] | join("\n")'
}

reconcile_duplicate_issue() {
  local canonical_issue_number="$1"
  local duplicate_issue_number="$2"
  local duplicate_issue_body="$3"
  local canonical_contents
  local duplicate_run_marker
  local migration_body
  local migration_marker="<!-- create-failure-issue-migrated:$duplicate_issue_number -->"

  canonical_contents="$(load_issue_contents "$canonical_issue_number")"
  duplicate_run_marker="$(
    grep -Eo '<!-- create-failure-issue-run:[^>]+ -->' <<<"$duplicate_issue_body" |
      head -n 1 || true
  )"

  if [[ "$canonical_contents" == *"$migration_marker"* ]] ||
    [[ -n "$duplicate_run_marker" && "$canonical_contents" == *"$duplicate_run_marker"* ]]; then
    echo "Issue #$duplicate_issue_number is already recorded on #$canonical_issue_number"
  else
    migration_body="$migration_marker

$duplicate_issue_body"
    echo "Recording issue #$duplicate_issue_number on canonical issue #$canonical_issue_number"
    gh issue comment "$canonical_issue_number" --body "$migration_body"
  fi

  gh issue close "$duplicate_issue_number" \
    --comment "Closing this concurrently-created alert as a duplicate of #$canonical_issue_number."
  echo "Closed duplicate issue #$duplicate_issue_number"
}

record_notification_once() {
  local issue_number="$1"
  local run_marker="$2"
  local comment_body="$3"
  local issue_contents

  issue_contents="$(load_issue_contents "$issue_number")"
  if [[ "$issue_contents" == *"$run_marker"* ]]; then
    echo "This workflow run is already recorded on issue #$issue_number"
    return
  fi

  echo "Adding this workflow run to issue #$issue_number"
  gh issue comment "$issue_number" --body "$comment_body"
  echo "Issue comment created successfully"
}

failed_jobs="$(
  jq -er '
    to_entries
    | map(select(.value.result == "failure"))
    | map(.key)
    | join(", ")
  ' <<<"$JOB_RESULTS"
)"

cancelled_jobs="$(
  jq -er '
    to_entries
    | map(select(.value.result == "cancelled"))
    | map(.key)
    | join(", ")
  ' <<<"$JOB_RESULTS"
)"

if [[ -z "$failed_jobs" && ( "$INCLUDE_CANCELLED" != "true" || -z "$cancelled_jobs" ) ]]; then
  echo "No reportable job failures or cancellations detected, skipping issue creation"
  exit 0
fi

if [[ -n "$failed_jobs" && ( "$INCLUDE_CANCELLED" != "true" || -z "$cancelled_jobs" ) ]]; then
  issue_title="$WORKFLOW_NAME Failed ($failed_jobs)"
  notification_body="The workflow **$WORKFLOW_NAME** failed during execution.

**Failed jobs:** $failed_jobs

**Run URL:** $RUN_URL

Please investigate the failed jobs and address any issues."
elif [[ -z "$failed_jobs" ]]; then
  issue_title="$WORKFLOW_NAME Cancelled ($cancelled_jobs)"
  notification_body="The workflow **$WORKFLOW_NAME** reported cancelled jobs.

**Cancelled jobs:** $cancelled_jobs

**Run URL:** $RUN_URL

Please investigate the cancelled jobs and address any issues."
else
  issue_title="$WORKFLOW_NAME Failed or Cancelled ($failed_jobs, $cancelled_jobs)"
  notification_body="The workflow **$WORKFLOW_NAME** reported failed or cancelled jobs.

**Failed jobs:** $failed_jobs

**Cancelled jobs:** $cancelled_jobs

**Run URL:** $RUN_URL

Please investigate the affected jobs and address any issues."
fi

issue_body="$notification_body"

if [[ "$DEDUPLICATE" == "true" ]]; then
  marker_key="$(jq -nr --arg workflow "$WORKFLOW_NAME" '$workflow | @uri')"
  deduplication_marker="<!-- create-failure-issue:$marker_key -->"
  run_marker_key="$(jq -nr --arg run_url "$RUN_URL" '$run_url | @uri')"
  run_marker="<!-- create-failure-issue-run:$run_marker_key -->"
  notification_comment="$run_marker

$notification_body"
  matching_issues="$(find_matching_issues "$deduplication_marker")"
  existing_issue_number="$(jq -r '.[0].number // empty' <<<"$matching_issues")"

  if [[ -n "$existing_issue_number" ]]; then
    echo "Found canonical issue #$existing_issue_number; reconciling open duplicates"
    while IFS= read -r duplicate_issue; do
      duplicate_issue_number="$(jq -r '.number' <<<"$duplicate_issue")"
      duplicate_issue_body="$(jq -r '.body // ""' <<<"$duplicate_issue")"
      reconcile_duplicate_issue \
        "$existing_issue_number" \
        "$duplicate_issue_number" \
        "$duplicate_issue_body"
    done < <(
      jq -c --argjson canonical "$existing_issue_number" \
        '.[] | select(.number != $canonical)' <<<"$matching_issues"
    )
    record_notification_once "$existing_issue_number" "$run_marker" "$notification_comment"
    exit 0
  fi

  issue_body="$deduplication_marker

$notification_comment"
fi

echo "Detected reportable job results; creating issue"
created_issue_url="$(
  gh issue create \
    --title "$issue_title" \
    --body "$issue_body" \
    --label ci
)"
echo "Issue created successfully: $created_issue_url"

if [[ "$DEDUPLICATE" == "true" ]]; then
  created_issue_number="${created_issue_url##*/}"
  if [[ ! "$created_issue_number" =~ ^[0-9]+$ ]]; then
    echo "Could not determine the created issue number from '$created_issue_url'" >&2
    exit 1
  fi

  # Different refs can pass the initial lookup concurrently. Keep the oldest
  # issue canonical and move this run there if this creation lost the race.
  matching_issues="$(find_matching_issues "$deduplication_marker")"
  canonical_issue_number="$(jq -r '.[0].number // empty' <<<"$matching_issues")"
  if [[ -n "$canonical_issue_number" && "$canonical_issue_number" != "$created_issue_number" ]]; then
    echo "Issue #$created_issue_number duplicates #$canonical_issue_number; reconciling"
    created_issue_body="$(
      jq -r --argjson created "$created_issue_number" \
        '.[] | select(.number == $created) | .body // ""' <<<"$matching_issues"
    )"
    if [[ -n "$created_issue_body" ]]; then
      reconcile_duplicate_issue \
        "$canonical_issue_number" \
        "$created_issue_number" \
        "$created_issue_body"
    fi
  fi
fi
