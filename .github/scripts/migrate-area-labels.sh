#!/usr/bin/env bash
#
# One-time migration to the A-* area-label convention (issue #6986).
#
# Maps the existing area-ish labels onto the A-* names used by
# .github/labeler-area.yml. GitHub preserves a label on all existing
# issues/PRs through a rename, so 1:1 renames are non-destructive. Merges
# (two old labels -> one A-* label) rename the higher-traffic label in place
# and re-tag the smaller label's items onto it before deleting it.
#
# Run once, manually, by a maintainer with triage rights, AFTER the auto-labeler
# is merged (actions/labeler applies existing labels but does not create them):
#
#   REPO=lance-format/lance ./.github/scripts/migrate-area-labels.sh
#
# Idempotent-ish: re-running skips steps whose target already exists.
#
# Not migrated (intentional):
#   - ci, documentation: kept as title-based conventional-commit labels that
#     feed release notes (.github/release.yml); A-ci / A-docs are separate
#     path-based area labels created fresh below.
#   - rust: left as-is; A-core was dropped, so core changes stay unlabelled.

set -euo pipefail

REPO="${REPO:-lance-format/lance}"
AREA_COLOR="006b75" # shared color for the A-* family

label_exists() {
  gh label list --repo "$REPO" --limit 500 --json name -q '.[].name' | grep -Fxq "$1"
}

# rename_label OLD NEW DESCRIPTION
rename_label() {
  local old="$1" new="$2" desc="$3"
  if label_exists "$new"; then
    echo "skip rename: '$new' already exists"
    return
  fi
  if ! label_exists "$old"; then
    echo "skip rename: source '$old' not found"
    return
  fi
  echo "rename '$old' -> '$new'"
  gh label edit "$old" --repo "$REPO" --name "$new" --description "$desc" --color "$AREA_COLOR"
}

# ensure_label NAME DESCRIPTION
ensure_label() {
  local name="$1" desc="$2"
  if label_exists "$name"; then
    echo "update '$name'"
    gh label edit "$name" --repo "$REPO" --description "$desc" --color "$AREA_COLOR"
    return
  fi
  echo "create '$name'"
  gh label create "$name" --repo "$REPO" --description "$desc" --color "$AREA_COLOR"
}

# merge_label SOURCE TARGET -- re-tag SOURCE's items onto TARGET, then delete SOURCE
merge_label() {
  local source="$1" target="$2"
  if ! label_exists "$source"; then
    echo "skip merge: source '$source' not found"
    return
  fi
  if ! label_exists "$target"; then
    echo "ERROR: merge target '$target' is missing; run the rename step first" >&2
    exit 1
  fi
  echo "merge '$source' -> '$target'"

  local issues prs
  issues=$(gh issue list --repo "$REPO" --label "$source" --state all --limit 1000 --json number -q '.[].number')
  prs=$(gh pr list --repo "$REPO" --label "$source" --state all --limit 1000 --json number -q '.[].number')

  if [ "$(printf '%s\n' "$issues" | grep -c .)" -ge 1000 ] || [ "$(printf '%s\n' "$prs" | grep -c .)" -ge 1000 ]; then
    echo "ERROR: '$source' has >=1000 items; re-tag manually to avoid truncation" >&2
    exit 1
  fi

  for n in $issues; do
    gh issue edit "$n" --repo "$REPO" --add-label "$target" >/dev/null
  done
  for n in $prs; do
    gh pr edit "$n" --repo "$REPO" --add-label "$target" >/dev/null
  done

  gh label delete "$source" --repo "$REPO" --yes
}

# --- 1:1 renames (non-destructive) ---
rename_label java "A-java" "Java bindings + JNI"
rename_label python "A-python" "Python bindings"

# --- merges: rename the higher-traffic label, fold in the smaller one ---
rename_label vector "A-index" "Vector index, linalg, tokenizer"
merge_label indexes "A-index"

rename_label file-storage "A-encoding" "Encoding, IO, file reader/writer"
rename_label format "A-format" "On-disk format: protos and format spec docs"

rename_label dependencies "A-deps" "Dependency updates"
merge_label "python:uv" "A-deps"

# --- brand-new labels (no existing source) ---
ensure_label "A-namespace" "Namespace impls"
ensure_label "A-docs" "Documentation"
ensure_label "A-ci" "CI / build workflows"

echo "Done. Current A-* labels:"
gh label list --repo "$REPO" --limit 500 --json name -q '.[].name' | grep '^A-' | sort
