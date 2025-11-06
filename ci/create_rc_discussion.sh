#!/bin/bash
set -e

# Script to create a GitHub Discussion for RC voting
# Usage: create_rc_discussion.sh <rc_tag> <rc_version> [release_branch] [release_type]
# Environment variables required: GH_TOKEN, GITHUB_REPOSITORY

RC_TAG=${1}
RC_VERSION=${2}
RELEASE_BRANCH=${3:-""}
RELEASE_TYPE=${4:-"minor"}  # major, minor, or patch

if [ -z "$RC_TAG" ] || [ -z "$RC_VERSION" ]; then
    echo "Error: RC_TAG and RC_VERSION are required"
    echo "Usage: create_rc_discussion.sh <rc_tag> <rc_version> [release_branch]"
    exit 1
fi

DISCUSSION_TITLE="[VOTE] Release Candidate ${RC_TAG}"

# Determine vote duration based on release type
case "$RELEASE_TYPE" in
    major)
        VOTE_DURATION_DAYS=7
        ;;
    minor)
        VOTE_DURATION_DAYS=3
        ;;
    patch)
        VOTE_DURATION_DAYS=0
        ;;
    *)
        VOTE_DURATION_DAYS=3
        ;;
esac

# Calculate vote end time in both UTC and Pacific
if [ "$VOTE_DURATION_DAYS" -gt 0 ]; then
    # Try macOS date format first, then GNU date format
    VOTE_END_TIME_UTC=$(date -u -v+${VOTE_DURATION_DAYS}d '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -u -d "+${VOTE_DURATION_DAYS} days" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "")
    VOTE_END_TIME_PT=$(TZ='America/Los_Angeles' date -v+${VOTE_DURATION_DAYS}d '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null || TZ='America/Los_Angeles' date -d "+${VOTE_DURATION_DAYS} days" '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null || echo "")
fi

# Build discussion body with testing instructions
DISCUSSION_BODY="## Release Candidate: ${RC_TAG}

This is a release candidate for version **${RC_VERSION}**.

### Release Information
- **RC Tag**: ${RC_TAG}"

if [ -n "$RELEASE_BRANCH" ]; then
    DISCUSSION_BODY="${DISCUSSION_BODY}
- **Release Branch**: ${RELEASE_BRANCH}"
fi

DISCUSSION_BODY="${DISCUSSION_BODY}
- **Release Notes**: https://github.com/lancedb/lance/releases/tag/${RC_TAG}

### Testing Instructions

#### Python
\`\`\`bash
pip install --pre --extra-index-url https://pypi.fury.io/lancedb/ pylance==${RC_VERSION}
\`\`\`

#### Java (Maven)
Add to your \`pom.xml\`:
\`\`\`xml
<dependency>
  <groupId>com.lancedb</groupId>
  <artifactId>lance</artifactId>
  <version>${RC_VERSION}</version>
</dependency>
\`\`\`

#### Rust (Cargo)
Add to your \`Cargo.toml\`:
\`\`\`toml
[dependencies]
lance = { version = \"=${RC_VERSION}\", git = \"https://github.com/lancedb/lance\", tag = \"${RC_TAG}\" }
\`\`\`

### Voting Instructions
Please test the RC artifacts and vote by commenting:
- **+1** to approve
- **0** to abstain or neutral
- **-1** if issues found (please include details)"

if [ "$VOTE_DURATION_DAYS" -gt 0 ] && [ -n "$VOTE_END_TIME_UTC" ]; then
    DISCUSSION_BODY="${DISCUSSION_BODY}

**Vote Duration**: If there are enough binding votes and no vetoes, the vote will end at **${VOTE_END_TIME_UTC} UTC**"

    if [ -n "$VOTE_END_TIME_PT" ]; then
        DISCUSSION_BODY="${DISCUSSION_BODY} (Pacific time: ${VOTE_END_TIME_PT})."
    else
        DISCUSSION_BODY="${DISCUSSION_BODY}."
    fi
else
    DISCUSSION_BODY="${DISCUSSION_BODY}

**Patch Release**: For patch releases, there is no duration requirement. The release will be cut as soon as there are enough binding votes and no vetoes."
fi

DISCUSSION_BODY="${DISCUSSION_BODY}

### Next Steps
- If approved: Approve RC using \`approve-rc\` workflow
- If issues found: Fix on release branch and create new RC using \`create-rc\` workflow"

# Get repository and category IDs using "Release Vote" category
REPO_DATA=$(gh api graphql -f query='
  query($owner: String!, $name: String!) {
    repository(owner: $owner, name: $name) {
      id
      discussionCategory(slug: "release-vote") {
        id
      }
    }
  }
' -f owner="$(echo ${GITHUB_REPOSITORY} | cut -d'/' -f1)" -f name="$(echo ${GITHUB_REPOSITORY} | cut -d'/' -f2)")

REPO_ID=$(echo "$REPO_DATA" | jq -r '.data.repository.id')
CATEGORY_ID=$(echo "$REPO_DATA" | jq -r '.data.repository.discussionCategory.id')

if [ -z "$CATEGORY_ID" ] || [ "$CATEGORY_ID" = "null" ]; then
    echo "Error: Discussion category 'Release Vote' not found. Please create it in repository settings."
    exit 1
fi

# Create discussion
DISCUSSION_URL=$(gh api graphql -f query='
  mutation($repositoryId: ID!, $categoryId: ID!, $body: String!, $title: String!) {
    createDiscussion(input: {repositoryId: $repositoryId, categoryId: $categoryId, body: $body, title: $title}) {
      discussion {
        url
      }
    }
  }
' -f repositoryId="$REPO_ID" -f categoryId="$CATEGORY_ID" \
  -f title="$DISCUSSION_TITLE" -f body="$DISCUSSION_BODY" \
  --jq '.data.createDiscussion.discussion.url')

echo "Created discussion: $DISCUSSION_URL" >&2
echo "$DISCUSSION_URL"
