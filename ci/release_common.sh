#!/bin/bash

# Common functions for release scripts

# Gets the current version from Cargo.toml
# Returns: version string (e.g., "1.3.0-beta.1")
get_version_from_cargo() {
    grep '^version = ' Cargo.toml | head -n1 | cut -d'"' -f2
}

# Parses version components from a version string
# Args: VERSION_STRING
# Returns: three values separated by spaces: MAJOR MINOR PATCH
# Example: parse_version_components "1.3.0-rc.2" returns "1 3 0"
parse_version_components() {
    local VERSION=$1
    local MAJOR=$(echo "${VERSION}" | cut -d. -f1 | sed 's/^v//')
    local MINOR=$(echo "${VERSION}" | cut -d. -f2)
    local PATCH=$(echo "${VERSION}" | cut -d. -f3 | cut -d- -f1)
    echo "${MAJOR} ${MINOR} ${PATCH}"
}

# Bumps version and commits the change
# Args: NEW_VERSION COMMIT_MESSAGE
bump_and_commit_version() {
    local NEW_VERSION=$1
    local COMMIT_MESSAGE=$2

    bump-my-version bump -vv --new-version "${NEW_VERSION}" --no-tag patch

    # Update Cargo.lock files after version bump
    cargo update
    (cd python && cargo update)
    (cd java/lance-jni && cargo update)

    git add -A
    git commit -m "${COMMIT_MESSAGE}"
}

# Determines the previous tag for release notes comparison
# Args: MAJOR MINOR PATCH [TAG_PREFIX]
# Returns: previous tag name or empty string
#
# For major/minor releases (PATCH=0):
#   - Checks for minor-release-root tag (minor release from release branch)
#   - Otherwise uses release-root tag (standard flow from main)
# For patch releases (PATCH>0):
#   - Compares against previous patch stable tag
determine_previous_tag() {
    local MAJOR=$1
    local MINOR=$2
    local PATCH=$3
    local TAG_PREFIX=${4:-"v"}

    if [ "${PATCH}" = "0" ]; then
        # Major/Minor release: check for minor-release-root tag first
        # This tag is created when a minor release is cut from a release branch
        local MINOR_RELEASE_ROOT_TAG="minor-release-root/${MAJOR}.${MINOR}.0"
        if git rev-parse "${MINOR_RELEASE_ROOT_TAG}" >/dev/null 2>&1; then
            # Read the source tag from the tag message
            local SOURCE_TAG=$(git tag -l --format='%(contents:subject)' "${MINOR_RELEASE_ROOT_TAG}")
            if [ -n "${SOURCE_TAG}" ]; then
                echo "${SOURCE_TAG}"
                return
            fi
        fi

        # Standard flow: use release-root tag
        local RELEASE_ROOT_TAG="release-root/${MAJOR}.${MINOR}.${PATCH}-beta.N"
        if git rev-parse "${RELEASE_ROOT_TAG}" >/dev/null 2>&1; then
            echo "${RELEASE_ROOT_TAG}"
        else
            echo ""
        fi
    else
        # Patch release: compare against previous stable tag
        local PREV_PATCH=$((PATCH - 1))
        local PREV_TAG="${TAG_PREFIX}${MAJOR}.${MINOR}.${PREV_PATCH}"
        if git rev-parse "${PREV_TAG}" >/dev/null 2>&1; then
            echo "${PREV_TAG}"
        else
            echo ""
        fi
    fi
}
