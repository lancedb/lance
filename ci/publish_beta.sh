#!/bin/bash
set -e

# Script to publish a beta preview release
# Usage: publish_beta.sh [branch_name]
# Example: publish_beta.sh main
# Example: publish_beta.sh release/v1.3

BRANCH=${1:-$(git branch --show-current)}
TAG_PREFIX=${2:-"v"}

readonly SELF_DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

echo "Publishing beta release from branch: ${BRANCH}"

# Ensure we're on the specified branch
git checkout "${BRANCH}"

# Read current version from Cargo.toml
CURRENT_VERSION=$(grep '^version = ' Cargo.toml | head -n1 | cut -d'"' -f2)
echo "Current version: ${CURRENT_VERSION}"

# Validate current version is a beta version
if [[ ! "${CURRENT_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+-beta\.[0-9]+$ ]]; then
    echo "ERROR: Current version ${CURRENT_VERSION} is not a beta version"
    echo "Expected format: X.Y.Z-beta.N"
    exit 1
fi

# Breaking change detection for main branch
if [[ "${BRANCH}" == "main" ]] && [[ "${CURRENT_VERSION}" =~ -beta\.[0-9]+$ ]]; then
    echo "Checking for breaking changes on main branch..."

    # Parse current version
    CURR_MAJOR=$(echo "${CURRENT_VERSION}" | cut -d. -f1)
    CURR_MINOR=$(echo "${CURRENT_VERSION}" | cut -d. -f2)
    CURR_PATCH=$(echo "${CURRENT_VERSION}" | cut -d. -f3 | cut -d- -f1)
    CURR_BETA=$(echo "${CURRENT_VERSION}" | sed 's/.*-beta\.//')

    # Find the release-root tag for the current version series
    CURR_RELEASE_ROOT_TAG="release-root/${CURR_MAJOR}.${CURR_MINOR}.${CURR_PATCH}-beta.N"

    if git rev-parse "${CURR_RELEASE_ROOT_TAG}" >/dev/null 2>&1; then
        echo "Found release root tag for current version: ${CURR_RELEASE_ROOT_TAG}"
        COMPARE_TAG="${CURR_RELEASE_ROOT_TAG}"
        COMPARE_COMMIT=$(git rev-parse "${CURR_RELEASE_ROOT_TAG}")
    else
        # No release-root tag found - skip breaking change detection for first time
        # But create the release-root tag at current HEAD for future comparisons
        echo "Release root tag ${CURR_RELEASE_ROOT_TAG} not found"
        echo "First time: skipping breaking change detection and creating release-root tag at current HEAD"
        echo "Future beta releases will compare against this tag"

        # We'll create the release-root tag after the beta increment below
        COMPARE_TAG=""
        COMPARE_COMMIT=""
        CREATE_INITIAL_RELEASE_ROOT="true"
    fi

    if [ -n "${COMPARE_TAG}" ]; then
        echo "Comparing against: ${COMPARE_TAG} (commit: ${COMPARE_COMMIT})"

        # Check for breaking changes
        BREAKING_CHANGES="false"
        if python3 "${SELF_DIR}/check_breaking_changes.py" --detect-only "${COMPARE_TAG}" "HEAD"; then
            echo "No breaking changes detected"
            BREAKING_CHANGES="false"
        else
            echo "Breaking changes detected"
            BREAKING_CHANGES="true"
        fi

        if [ "${BREAKING_CHANGES}" = "true" ]; then
            # Extract base RC version from release-root tag message
            TAG_MESSAGE=$(git tag -l --format='%(contents)' "${CURR_RELEASE_ROOT_TAG}")
            BASE_RC_VERSION=$(echo "${TAG_MESSAGE}" | head -n1 | sed 's/Base: //')
            BASE_VERSION=$(echo "${BASE_RC_VERSION}" | sed 's/-rc\.[0-9]*$//')
            BASE_MAJOR=$(echo "${BASE_VERSION}" | cut -d. -f1)

            echo "Base RC version: ${BASE_RC_VERSION} (major: ${BASE_MAJOR})"

            # Check if major already bumped from base
            if [ "${CURR_MAJOR}" -gt "${BASE_MAJOR}" ]; then
                echo "Breaking changes exist, but major version already bumped from ${BASE_MAJOR} to ${CURR_MAJOR}"
                echo "No additional major version bump needed"
            else
                echo "Breaking changes detected since ${BASE_VERSION}, bumping major version"
                NEXT_MAJOR=$((CURR_MAJOR + 1))
                NEXT_VERSION="${NEXT_MAJOR}.0.0-beta.1"
                echo "Bumping to ${NEXT_VERSION}"

                echo "Updating version from ${CURRENT_VERSION} to ${NEXT_VERSION}"
                bump-my-version bump -vv --new-version "${NEXT_VERSION}" --no-tag patch

                # Update Cargo.lock files after version bump
                cargo update
                (cd python && cargo update)
                (cd java/lance-jni && cargo update)

                git add -A
                git commit -m "chore: bump to ${NEXT_VERSION} based on breaking change detection"

                CURRENT_VERSION="${NEXT_VERSION}"

                # Create new release-root tag pointing to same commit (same base for comparison)
                NEW_RELEASE_ROOT_TAG="release-root/${NEXT_MAJOR}.0.0-beta.N"
                if git rev-parse "${NEW_RELEASE_ROOT_TAG}" >/dev/null 2>&1; then
                    echo "Release root tag ${NEW_RELEASE_ROOT_TAG} already exists"
                else
                    echo "Creating new release root tag: ${NEW_RELEASE_ROOT_TAG} pointing to commit ${COMPARE_COMMIT}"
                    git tag -a "${NEW_RELEASE_ROOT_TAG}" "${COMPARE_COMMIT}" -m "Base: ${BASE_RC_VERSION}
Release root for ${NEXT_MAJOR}.0.0-beta.N series (same base as ${CURR_MAJOR}.${CURR_MINOR}.${CURR_PATCH}-beta.N)"
                fi
                BETA_TAG="${TAG_PREFIX}${CURRENT_VERSION}"
                echo "Creating beta tag: ${BETA_TAG}"
                git tag -a "${BETA_TAG}" -m "Beta release version ${CURRENT_VERSION}"

                echo "Successfully published beta release: ${BETA_TAG}"
                echo "BETA_TAG=${BETA_TAG}" >> $GITHUB_OUTPUT 2>/dev/null || true
                echo "BETA_VERSION=${CURRENT_VERSION}" >> $GITHUB_OUTPUT 2>/dev/null || true
                echo "RELEASE_ROOT_TAG=${NEW_RELEASE_ROOT_TAG}" >> $GITHUB_OUTPUT 2>/dev/null || true
                echo "RELEASE_NOTES_FROM=${NEW_RELEASE_ROOT_TAG}" >> $GITHUB_OUTPUT 2>/dev/null || true
                exit 0
            fi
        fi
    else
        echo "Warning: No compare tag found for breaking change detection"
    fi
fi

# Bump beta version (beta.N → beta.N+1)
echo "Bumping beta version"
bump-my-version bump -vv prerelease_num

# Update Cargo.lock files after version bump
cargo update
(cd python && cargo update)
(cd java/lance-jni && cargo update)

# Get new version
NEW_VERSION=$(grep '^version = ' Cargo.toml | head -n1 | cut -d'"' -f2)
echo "New version: ${NEW_VERSION}"

# Commit the version change
git add -A
git commit -m "chore: release beta version ${NEW_VERSION}"

# Create beta tag
BETA_TAG="${TAG_PREFIX}${NEW_VERSION}"
echo "Creating beta tag: ${BETA_TAG}"
git tag -a "${BETA_TAG}" -m "Beta release version ${NEW_VERSION}"

# Create initial release-root tag if this is the first time
CREATED_RELEASE_ROOT_TAG=""
if [ "${CREATE_INITIAL_RELEASE_ROOT:-false}" = "true" ]; then
    BETA_MAJOR=$(echo "${NEW_VERSION}" | cut -d. -f1)
    BETA_MINOR=$(echo "${NEW_VERSION}" | cut -d. -f2)
    BETA_PATCH=$(echo "${NEW_VERSION}" | cut -d. -f3 | cut -d- -f1)
    INITIAL_RELEASE_ROOT_TAG="release-root/${BETA_MAJOR}.${BETA_MINOR}.${BETA_PATCH}-beta.N"

    echo "Creating initial release-root tag: ${INITIAL_RELEASE_ROOT_TAG} at HEAD"
    git tag -a "${INITIAL_RELEASE_ROOT_TAG}" "HEAD" -m "Base: ${NEW_VERSION}
Release root for ${BETA_MAJOR}.${BETA_MINOR}.${BETA_PATCH}-beta.N series (initial)"
    echo "Created initial release-root tag for future breaking change detection"
    CREATED_RELEASE_ROOT_TAG="${INITIAL_RELEASE_ROOT_TAG}"
fi

# Determine release notes comparison base
BETA_MAJOR=$(echo "${NEW_VERSION}" | cut -d. -f1)
BETA_MINOR=$(echo "${NEW_VERSION}" | cut -d. -f2)
BETA_PATCH=$(echo "${NEW_VERSION}" | cut -d. -f3 | cut -d- -f1)

if [[ "${BRANCH}" == "main" ]]; then
    # For main branch: compare against release-root tag
    BETA_RELEASE_ROOT_TAG="release-root/${BETA_MAJOR}.${BETA_MINOR}.${BETA_PATCH}-beta.N"

    if git rev-parse "${BETA_RELEASE_ROOT_TAG}" >/dev/null 2>&1; then
        echo "Release notes will compare from ${BETA_RELEASE_ROOT_TAG} to ${BETA_TAG}"
        RELEASE_NOTES_FROM="${BETA_RELEASE_ROOT_TAG}"
    else
        echo "Warning: Release root tag ${BETA_RELEASE_ROOT_TAG} not found"
        RELEASE_NOTES_FROM=""
    fi
elif [[ "${BRANCH}" =~ ^release/ ]]; then
    # For release branch: compare against last stable tag
    PREV_PATCH=$((BETA_PATCH - 1))
    if [ "${PREV_PATCH}" -ge 0 ]; then
        PREV_STABLE_TAG="${TAG_PREFIX}${BETA_MAJOR}.${BETA_MINOR}.${PREV_PATCH}"
        if git rev-parse "${PREV_STABLE_TAG}" >/dev/null 2>&1; then
            echo "Release notes will compare from ${PREV_STABLE_TAG} to ${BETA_TAG}"
            RELEASE_NOTES_FROM="${PREV_STABLE_TAG}"
        else
            echo "Warning: Previous stable tag ${PREV_STABLE_TAG} not found"
            RELEASE_NOTES_FROM=""
        fi
    else
        echo "Warning: No previous patch to compare against (patch is 0)"
        RELEASE_NOTES_FROM=""
    fi
else
    RELEASE_NOTES_FROM=""
fi

echo "Successfully published beta release: ${BETA_TAG}"
echo "BETA_TAG=${BETA_TAG}" >> $GITHUB_OUTPUT 2>/dev/null || true
echo "BETA_VERSION=${NEW_VERSION}" >> $GITHUB_OUTPUT 2>/dev/null || true
echo "RELEASE_NOTES_FROM=${RELEASE_NOTES_FROM}" >> $GITHUB_OUTPUT 2>/dev/null || true
echo "RELEASE_ROOT_TAG=${CREATED_RELEASE_ROOT_TAG}" >> $GITHUB_OUTPUT 2>/dev/null || true
