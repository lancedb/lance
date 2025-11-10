#!/bin/bash
set -e

# Script to create a release branch with initial RC for major/minor release
# Always creates RC from the tip of main branch
# Checks for breaking changes and bumps major version if needed
# The version is automatically determined from main branch HEAD
# Usage: create_release_branch.sh
# Example: create_release_branch.sh

TAG_PREFIX=${1:-"v"}

readonly SELF_DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

git checkout main
MAIN_VERSION=$(grep '^version = ' Cargo.toml | head -n1 | cut -d'"' -f2)
echo "Main branch current version: ${MAIN_VERSION}"

# Extract the base version from main (remove beta suffix if present)
if [[ "${MAIN_VERSION}" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)(-beta\.([0-9]+))?$ ]]; then
    CURR_MAJOR="${BASH_REMATCH[1]}"
    CURR_MINOR="${BASH_REMATCH[2]}"
    CURR_PATCH="${BASH_REMATCH[3]}"
    BASE_VERSION="${CURR_MAJOR}.${CURR_MINOR}.${CURR_PATCH}"
else
    echo "ERROR: Cannot parse version from main branch: ${MAIN_VERSION}"
    exit 1
fi

echo "Current base version on main: ${BASE_VERSION}"

# Check for existing release-root tag to find comparison base
CURR_RELEASE_ROOT_TAG="release-root/${BASE_VERSION}-beta.N"

if git rev-parse "${CURR_RELEASE_ROOT_TAG}" >/dev/null 2>&1; then
    echo "Found release root tag: ${CURR_RELEASE_ROOT_TAG}"
    COMPARE_TAG="${CURR_RELEASE_ROOT_TAG}"
    COMPARE_COMMIT=$(git rev-parse "${CURR_RELEASE_ROOT_TAG}")
    echo "Will compare against: ${COMPARE_TAG} (commit: ${COMPARE_COMMIT})"
else
    echo "No release root tag found for current version series"
    COMPARE_TAG=""
fi

# Check for breaking changes
BREAKING_CHANGES="false"
if [ -n "${COMPARE_TAG}" ]; then
    if python3 "${SELF_DIR}/check_breaking_changes.py" --detect-only "${COMPARE_TAG}" "HEAD"; then
        echo "No breaking changes detected"
        BREAKING_CHANGES="false"
    else
        echo "Breaking changes detected"
        BREAKING_CHANGES="true"
    fi
fi

# Determine RC version based on breaking changes
if [ "${BREAKING_CHANGES}" = "true" ]; then
    # Extract base RC version from release-root tag message
    TAG_MESSAGE=$(git tag -l --format='%(contents)' "${CURR_RELEASE_ROOT_TAG}")
    BASE_RC_VERSION=$(echo "${TAG_MESSAGE}" | head -n1 | sed 's/Base: //')
    BASE_RC_MAJOR=$(echo "${BASE_RC_VERSION}" | cut -d. -f1 | sed 's/^v//')

    echo "Base RC version: ${BASE_RC_VERSION} (major: ${BASE_RC_MAJOR})"

    if [ "${CURR_MAJOR}" -gt "${BASE_RC_MAJOR}" ]; then
        echo "Major version already bumped from ${BASE_RC_MAJOR} to ${CURR_MAJOR}"
        RC_VERSION="${BASE_VERSION}-rc.1"
    else
        echo "Breaking changes require major version bump"
        RC_MAJOR=$((CURR_MAJOR + 1))
        RC_VERSION="${RC_MAJOR}.0.0-rc.1"
    fi
else
    # No breaking changes, use current base version
    RC_VERSION="${BASE_VERSION}-rc.1"
fi

echo "Creating RC version: ${RC_VERSION}"

# Determine release type (major if X.0.0, otherwise minor)
RC_MINOR=$(echo "${RC_VERSION}" | cut -d. -f2 | cut -d- -f1)
if [ "${RC_MINOR}" = "0" ]; then
    RELEASE_TYPE="major"
else
    RELEASE_TYPE="minor"
fi
echo "Release type: ${RELEASE_TYPE}"

# Parse RC version for release branch
RC_MAJOR=$(echo "${RC_VERSION}" | cut -d. -f1)
RC_MINOR=$(echo "${RC_VERSION}" | cut -d. -f2)
RELEASE_BRANCH="release/v${RC_MAJOR}.${RC_MINOR}"

echo "Will create release branch: ${RELEASE_BRANCH}"

# Create release branch from main HEAD
echo "Creating release branch ${RELEASE_BRANCH} from main HEAD"
git checkout -b "${RELEASE_BRANCH}"

# Set version to RC version
echo "Setting version to ${RC_VERSION}"
bump-my-version bump -vv --new-version "${RC_VERSION}" --no-tag patch

# Update Cargo.lock files after version bump
cargo update
(cd python && cargo update)
(cd java/lance-jni && cargo update)

# Commit the RC version
git add -A
git commit -m "chore: release candidate ${RC_VERSION}"

# Create the RC tag
RC_TAG="${TAG_PREFIX}${RC_VERSION}"
echo "Creating tag ${RC_TAG}"
git tag -a "${RC_TAG}" -m "Release candidate ${RC_VERSION}"

echo "Successfully created RC tag: ${RC_TAG} on branch ${RELEASE_BRANCH}"

# Now bump main to next unreleased version (beta.0)
echo "Bumping main to next version beta.0"
git checkout main

# Determine next version for main based on RC version
# Always bump minor from the RC version
NEXT_MAJOR="${RC_MAJOR}"
NEXT_MINOR=$((RC_MINOR + 1))
NEXT_VERSION="${NEXT_MAJOR}.${NEXT_MINOR}.0-beta.0"

echo "Bumping main to ${NEXT_VERSION} (unreleased)"

bump-my-version bump -vv --new-version "${NEXT_VERSION}" --no-tag patch

# Update Cargo.lock files after version bump
cargo update
(cd python && cargo update)
(cd java/lance-jni && cargo update)

git add -A
git commit -m "chore: bump main to ${NEXT_VERSION}

Unreleased version after creating ${RC_TAG}"

echo "Main branch bumped to ${NEXT_VERSION}"

# Create release-root tag for the new beta series on main (points to commit before RC branch)
# Strip the prerelease suffix from NEXT_VERSION for the tag name
NEXT_BASE_VERSION="${NEXT_MAJOR}.${NEXT_MINOR}.0"
RELEASE_ROOT_TAG="release-root/${NEXT_BASE_VERSION}-beta.N"
echo "Creating release root tag ${RELEASE_ROOT_TAG} pointing to RC ${RC_VERSION}"
git tag -a "${RELEASE_ROOT_TAG}" "${RC_TAG}^" -m "Base: ${RC_VERSION}
Release root for ${NEXT_BASE_VERSION}-beta.N series"

# Determine comparison base for RC release notes
# For major/minor RC, we want to compare against the OLD release-root tag (the one for the main version before bump)
# which points to the previous RC base
OLD_RELEASE_ROOT_TAG="release-root/${BASE_VERSION}-beta.N"

if git rev-parse "${OLD_RELEASE_ROOT_TAG}" >/dev/null 2>&1; then
    PREVIOUS_TAG="${OLD_RELEASE_ROOT_TAG}"
    echo "Release notes will compare against previous release-root: ${PREVIOUS_TAG}"
else
    echo "Warning: Release root tag ${OLD_RELEASE_ROOT_TAG} not found"
    PREVIOUS_TAG=""
fi

# Output for GitHub Actions
echo "RC_TAG=${RC_TAG}" >> $GITHUB_OUTPUT 2>/dev/null || true
echo "RC_VERSION=${RC_VERSION}" >> $GITHUB_OUTPUT 2>/dev/null || true
echo "RELEASE_BRANCH=${RELEASE_BRANCH}" >> $GITHUB_OUTPUT 2>/dev/null || true
echo "MAIN_VERSION=${NEXT_VERSION}" >> $GITHUB_OUTPUT 2>/dev/null || true
echo "RELEASE_ROOT_TAG=${RELEASE_ROOT_TAG}" >> $GITHUB_OUTPUT 2>/dev/null || true
echo "PREVIOUS_TAG=${PREVIOUS_TAG}" >> $GITHUB_OUTPUT 2>/dev/null || true
echo "RELEASE_TYPE=${RELEASE_TYPE}" >> $GITHUB_OUTPUT 2>/dev/null || true

echo "Successfully created major/minor RC!"
echo "  RC Tag: ${RC_TAG}"
echo "  Release Branch: ${RELEASE_BRANCH}"
echo "  Main Version: ${NEXT_VERSION}"
echo "  Release Root Tag: ${RELEASE_ROOT_TAG}"
