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

if [[ "$DEDUPLICATE" == "true" ]]; then
  marker_key="$(jq -nr --arg workflow "$WORKFLOW_NAME" '$workflow | @uri')"
  deduplication_marker="<!-- create-failure-issue:$marker_key -->"
  existing_issue_number="$({
    gh issue list \
      --state open \
      --label ci \
      --limit 1000 \
      --json number,body
  } | jq -r --arg marker "$deduplication_marker" '
    first(.[] | select((.body // "") | contains($marker)))
    | .number // empty
  ')"

  if [[ -n "$existing_issue_number" ]]; then
    echo "Found existing issue #$existing_issue_number; adding this run as a comment"
    gh issue comment "$existing_issue_number" --body "$notification_body"
    echo "Issue comment created successfully"
    exit 0
  fi

  notification_body="$deduplication_marker

$notification_body"
fi

echo "Detected reportable job results; creating issue"
gh issue create \
  --title "$issue_title" \
  --body "$notification_body" \
  --label ci
echo "Issue created successfully"
