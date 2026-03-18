#!/bin/bash
set -e

# Script to create a release branch with initial RC for major/minor release
# Can create from main branch or from an existing release branch
#
# Usage: create_release_branch.sh [source_release_branch] [tag_prefix]
#
# Examples:
#   create_release_branch.sh                    # Create from main branch
#   create_release_branch.sh release/v1.3      # Create minor release from release/v1.3
#   create_release_branch.sh "" v              # Create from main with custom prefix

SOURCE_RELEASE_BRANCH=${1:-""}
TAG_PREFIX=${2:-"v"}

readonly SELF_DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Source common release functions
source "${SELF_DIR}/release_common.sh"

# Determine if we're creating from main or from a release branch
if [ -n "${SOURCE_RELEASE_BRANCH}" ]; then
    echo "Creating minor release from release branch: ${SOURCE_RELEASE_BRANCH}"
    CREATE_FROM_RELEASE_BRANCH="true"
else
    echo "Creating release from main branch"
    CREATE_FROM_RELEASE_BRANCH="false"
fi

# Always check main version first (for validation when creating from release branch)
git fetch origin main
MAIN_VERSION=$(git show origin/main:Cargo.toml | grep '^version = ' | head -n1 | cut -d'"' -f2)
echo "Main branch version: ${MAIN_VERSION}"

# Parse main version
if [[ "${MAIN_VERSION}" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)(-beta\.([0-9]+))?$ ]]; then
    MAIN_MAJOR="${BASH_REMATCH[1]}"
    MAIN_MINOR="${BASH_REMATCH[2]}"
    MAIN_PATCH="${BASH_REMATCH[3]}"
else
    echo "ERROR: Cannot parse version from main branch: ${MAIN_VERSION}"
    exit 1
fi

if [ "${CREATE_FROM_RELEASE_BRANCH}" = "true" ]; then
    #
    # ============= CREATE FROM RELEASE BRANCH =============
    #
    # Validate main is at a major version (X.0.0-beta.N)
    if [ "${MAIN_MINOR}" != "0" ] || [ "${MAIN_PATCH}" != "0" ]; then
        echo "ERROR: Cannot create minor release from release branch when main is not at a major version"
        echo "Main is at ${MAIN_VERSION}, expected X.0.0-beta.N format"
        echo "Minor releases from release branches are only allowed when main is targeting a major release"
        exit 1
    fi

    echo "Main is at major version ${MAIN_MAJOR}.0.0 - OK to create minor release from release branch"

    # Checkout the source release branch
    git checkout "${SOURCE_RELEASE_BRANCH}"
    SOURCE_VERSION=$(get_version_from_cargo)
    echo "Source release branch version: ${SOURCE_VERSION}"

    # Parse source version
    if [[ "${SOURCE_VERSION}" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)(-beta\.([0-9]+))?$ ]]; then
        SOURCE_MAJOR="${BASH_REMATCH[1]}"
        SOURCE_MINOR="${BASH_REMATCH[2]}"
        SOURCE_PATCH="${BASH_REMATCH[3]}"
    else
        echo "ERROR: Cannot parse version from source branch: ${SOURCE_VERSION}"
        exit 1
    fi

    # Validate source branch is in the same major version series (or one less than main)
    if [ "${SOURCE_MAJOR}" -ge "${MAIN_MAJOR}" ]; then
        echo "ERROR: Source branch major version (${SOURCE_MAJOR}) must be less than main major version (${MAIN_MAJOR})"
        exit 1
    fi

    # Determine next minor version
    RC_MAJOR="${SOURCE_MAJOR}"
    RC_MINOR=$((SOURCE_MINOR + 1))
    RC_VERSION="${RC_MAJOR}.${RC_MINOR}.0-rc.1"

    echo "Creating RC version: ${RC_VERSION}"

    # Release type is always minor when creating from release branch
    RELEASE_TYPE="minor"
    echo "Release type: ${RELEASE_TYPE}"

    # Create new release branch from source branch
    RELEASE_BRANCH="release/v${RC_MAJOR}.${RC_MINOR}"
    echo "Creating release branch ${RELEASE_BRANCH} from ${SOURCE_RELEASE_BRANCH}"
    git checkout -b "${RELEASE_BRANCH}"

    # Set version to RC version
    echo "Setting version to ${RC_VERSION}"
    bump_and_commit_version "${RC_VERSION}" "chore: release candidate ${RC_VERSION}

Created from ${SOURCE_RELEASE_BRANCH}"

    # Create the RC tag
    RC_TAG="${TAG_PREFIX}${RC_VERSION}"
    echo "Creating tag ${RC_TAG}"
    git tag -a "${RC_TAG}" -m "Release candidate ${RC_VERSION}

Created from ${SOURCE_RELEASE_BRANCH}"

    echo "Successfully created RC tag: ${RC_TAG} on branch ${RELEASE_BRANCH}"

    # Find latest stable tag on source branch for release notes comparison
    # Look for tags matching vX.Y.* where X.Y matches source branch
    LATEST_STABLE_TAG=$(git tag -l "${TAG_PREFIX}${SOURCE_MAJOR}.${SOURCE_MINOR}.*" | grep -v -E '(beta|rc)' | sort -V | tail -n1)

    if [ -n "${LATEST_STABLE_TAG}" ]; then
        PREVIOUS_TAG="${LATEST_STABLE_TAG}"
        echo "Release notes will compare against latest stable: ${PREVIOUS_TAG}"

        # Create minor-release-root tag to mark this as a minor release from a release branch
        # This tag stores the source stable tag for use by determine_previous_tag
        MINOR_RELEASE_ROOT_TAG="minor-release-root/${RC_MAJOR}.${RC_MINOR}.0"
        echo "Creating minor release root tag: ${MINOR_RELEASE_ROOT_TAG}"
        git tag -a "${MINOR_RELEASE_ROOT_TAG}" -m "${PREVIOUS_TAG}"
    else
        echo "Warning: No stable tag found for ${SOURCE_MAJOR}.${SOURCE_MINOR}.* series"
        PREVIOUS_TAG=""
    fi

    # Output for GitHub Actions (no main version or release root tag when creating from release branch)
    echo "RC_TAG=${RC_TAG}" >> $GITHUB_OUTPUT 2>/dev/null || true
    echo "RC_VERSION=${RC_VERSION}" >> $GITHUB_OUTPUT 2>/dev/null || true
    echo "RELEASE_BRANCH=${RELEASE_BRANCH}" >> $GITHUB_OUTPUT 2>/dev/null || true
    echo "PREVIOUS_TAG=${PREVIOUS_TAG}" >> $GITHUB_OUTPUT 2>/dev/null || true
    echo "RELEASE_TYPE=${RELEASE_TYPE}" >> $GITHUB_OUTPUT 2>/dev/null || true
    echo "SOURCE_RELEASE_BRANCH=${SOURCE_RELEASE_BRANCH}" >> $GITHUB_OUTPUT 2>/dev/null || true
    echo "MINOR_RELEASE_ROOT_TAG=${MINOR_RELEASE_ROOT_TAG:-}" >> $GITHUB_OUTPUT 2>/dev/null || true

    echo "Successfully created minor RC from release branch!"
    echo "  RC Tag: ${RC_TAG}"
    echo "  Release Branch: ${RELEASE_BRANCH}"
    echo "  Source Branch: ${SOURCE_RELEASE_BRANCH}"
    echo "  Release Notes Base: ${PREVIOUS_TAG}"
    echo "  Minor Release Root Tag: ${MINOR_RELEASE_ROOT_TAG:-none}"

else
    #
    # ============= CREATE FROM MAIN BRANCH =============
    #
    git checkout main
    BASE_VERSION="${MAIN_MAJOR}.${MAIN_MINOR}.${MAIN_PATCH}"
    CURR_MAJOR="${MAIN_MAJOR}"
    CURR_MINOR="${MAIN_MINOR}"
    CURR_PATCH="${MAIN_PATCH}"

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
fi
