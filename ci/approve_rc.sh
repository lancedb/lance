#!/bin/bash
set -e

# Script to approve RC and promote to stable release
# Works for both major/minor and patch releases
# Usage: approve_rc.sh <rc_tag>
# Example: approve_rc.sh v1.3.0-rc.2

RC_TAG=${1:?"Error: RC tag required (e.g., v1.3.0-rc.2)"}
TAG_PREFIX=${2:-"v"}

readonly SELF_DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Source common release functions
source "${SELF_DIR}/release_common.sh"

echo "Promoting RC tag ${RC_TAG} to stable release"

# Parse version from RC tag (v1.3.0-rc.2 → 1.3.0)
RC_VERSION=$(echo "${RC_TAG}" | sed "s/^${TAG_PREFIX}//")
STABLE_VERSION=$(echo "${RC_VERSION}" | sed 's/-rc\.[0-9]*$//')

echo "Stable version will be: ${STABLE_VERSION}"

# Parse major.minor.patch
read MAJOR MINOR PATCH <<< $(parse_version_components "${STABLE_VERSION}")
RELEASE_BRANCH="release/v${MAJOR}.${MINOR}"

echo "Release branch: ${RELEASE_BRANCH}"

# Checkout release branch
git checkout "${RELEASE_BRANCH}"

# Verify we're at the correct RC version
CURRENT_VERSION=$(get_version_from_cargo)
if [ "${CURRENT_VERSION}" != "${RC_VERSION}" ]; then
    echo "ERROR: Branch is at ${CURRENT_VERSION}, expected ${RC_VERSION}"
    echo "Make sure the RC tag matches the branch state"
    exit 1
fi

# Bump from RC to stable
echo "Bumping version from ${RC_VERSION} to ${STABLE_VERSION}"
bump_and_commit_version "${STABLE_VERSION}" "chore: release version ${STABLE_VERSION}

Promoted from ${RC_TAG}"

# Create stable tag
STABLE_TAG="${TAG_PREFIX}${STABLE_VERSION}"
echo "Creating stable tag: ${STABLE_TAG}"
git tag -a "${STABLE_TAG}" -m "Release version ${STABLE_VERSION}"

# Determine if this is a major/minor release or patch release
if [ "${PATCH}" = "0" ]; then
    echo "This is a major/minor release (${STABLE_VERSION})"
    IS_MAJOR_MINOR="true"
else
    echo "This is a patch release (${STABLE_VERSION})"
    IS_MAJOR_MINOR="false"
fi

# Determine previous tag for release notes
PREVIOUS_TAG=$(determine_previous_tag "${MAJOR}" "${MINOR}" "${PATCH}" "${TAG_PREFIX}")
if [ -n "${PREVIOUS_TAG}" ]; then
    echo "Release notes will compare against: ${PREVIOUS_TAG}"
else
    echo "Warning: Previous tag not found"
fi

# Always auto-bump to next patch beta.0 after stable release
NEXT_PATCH=$((PATCH + 1))
NEXT_BETA_VERSION="${MAJOR}.${MINOR}.${NEXT_PATCH}-beta.0"

echo "Bumping to ${NEXT_BETA_VERSION} for next patch development"
bump_and_commit_version "${NEXT_BETA_VERSION}" "chore: bump to ${NEXT_BETA_VERSION} for next patch development"

echo "Successfully promoted to stable release: ${STABLE_TAG}"
echo "Release branch bumped to ${NEXT_BETA_VERSION}"

echo "STABLE_TAG=${STABLE_TAG}" >> $GITHUB_OUTPUT 2>/dev/null || true
echo "STABLE_VERSION=${STABLE_VERSION}" >> $GITHUB_OUTPUT 2>/dev/null || true
echo "RELEASE_BRANCH=${RELEASE_BRANCH}" >> $GITHUB_OUTPUT 2>/dev/null || true
echo "IS_MAJOR_MINOR=${IS_MAJOR_MINOR}" >> $GITHUB_OUTPUT 2>/dev/null || true
echo "NEXT_BETA_VERSION=${NEXT_BETA_VERSION}" >> $GITHUB_OUTPUT 2>/dev/null || true
echo "PREVIOUS_TAG=${PREVIOUS_TAG}" >> $GITHUB_OUTPUT 2>/dev/null || true
