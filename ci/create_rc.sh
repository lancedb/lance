#!/bin/bash
set -e

# Script to create RC on an existing release branch
# Works for patch rc.1 and iteration rc.2, rc.3, etc.
# Usage: create_rc.sh <release_branch>
# Example: create_rc.sh release/v1.3

RELEASE_BRANCH=${1:?"Error: release branch required (e.g., release/v1.3)"}
TAG_PREFIX=${2:-"v"}

readonly SELF_DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Source common release functions
source "${SELF_DIR}/release_common.sh"

echo "Creating RC on release branch: ${RELEASE_BRANCH}"

# Checkout release branch
git checkout "${RELEASE_BRANCH}"

# Read current version from Cargo.toml
CURRENT_VERSION=$(get_version_from_cargo)
echo "Current version on branch: ${CURRENT_VERSION}"

# Validate version format - should be beta.N or rc.N
if [[ "${CURRENT_VERSION}" =~ ^([0-9]+\.[0-9]+\.[0-9]+)-beta\.([0-9]+)$ ]]; then
    # At beta version, determine next RC number
    BASE_VERSION="${BASH_REMATCH[1]}"

    # Find highest RC tag for this base version
    HIGHEST_RC=$(git tag -l "${TAG_PREFIX}${BASE_VERSION}-rc.*" | sed "s/^${TAG_PREFIX}${BASE_VERSION}-rc\.//" | sort -n | tail -n1)

    if [ -z "${HIGHEST_RC}" ]; then
        # No RC exists yet, start with rc.1
        RC_NUMBER=1
    else
        # Increment the highest RC
        RC_NUMBER=$((HIGHEST_RC + 1))
    fi

    RC_VERSION="${BASE_VERSION}-rc.${RC_NUMBER}"

elif [[ "${CURRENT_VERSION}" =~ ^([0-9]+\.[0-9]+\.[0-9]+)-rc\.([0-9]+)$ ]]; then
    # At rc.N version, increment RC number
    BASE_VERSION="${BASH_REMATCH[1]}"
    CURRENT_RC="${BASH_REMATCH[2]}"
    RC_NUMBER=$((CURRENT_RC + 1))
    RC_VERSION="${BASE_VERSION}-rc.${RC_NUMBER}"
elif [[ "${CURRENT_VERSION}" =~ ^([0-9]+\.[0-9]+\.[0-9]+)$ ]]; then
    # At stable version - this shouldn't happen as approve-rc workflow auto-bumps to beta.0
    echo "ERROR: Release branch is at stable version ${CURRENT_VERSION}"
    echo "Expected format: X.Y.Z-beta.N or X.Y.Z-rc.N"
    echo "The release branch should have been auto-bumped to beta.0 after RC approval"
    exit 1
else
    echo "ERROR: Unexpected version format: ${CURRENT_VERSION}"
    echo "Expected format: X.Y.Z-beta.N or X.Y.Z-rc.N"
    exit 1
fi

echo "Creating RC version: ${RC_VERSION}"
bump_and_commit_version "${RC_VERSION}" "chore: release candidate ${RC_VERSION}"

# Create the RC tag
RC_TAG="${TAG_PREFIX}${RC_VERSION}"
echo "Creating tag ${RC_TAG}"
git tag -a "${RC_TAG}" -m "Release candidate ${RC_VERSION}"

# Determine comparison base for release notes
read MAJOR MINOR PATCH <<< $(parse_version_components "${BASE_VERSION}")

# Determine previous tag for release notes
PREVIOUS_TAG=$(determine_previous_tag "${MAJOR}" "${MINOR}" "${PATCH}" "${TAG_PREFIX}")
if [ -n "${PREVIOUS_TAG}" ]; then
    echo "Release notes will compare against: ${PREVIOUS_TAG}"
else
    echo "Warning: Previous tag not found"
fi

echo "Successfully created RC tag: ${RC_TAG}"
echo "RC_TAG=${RC_TAG}" >> $GITHUB_OUTPUT 2>/dev/null || true
echo "RC_VERSION=${RC_VERSION}" >> $GITHUB_OUTPUT 2>/dev/null || true
echo "PREVIOUS_TAG=${PREVIOUS_TAG}" >> $GITHUB_OUTPUT 2>/dev/null || true
echo "RELEASE_TYPE=patch" >> $GITHUB_OUTPUT 2>/dev/null || true
