# Lance Community Voting Process

Lance uses a consensus-based voting process for decision-making.

## Expressing Votes

Votes are expressed as the following:

- **+1**: Yes
- **0**: Abstain
- **-1**: No

When voting, it is recommended that voters indicate whether their vote is binding or not (e.g., `+1 (non-binding)`, `-1 (binding)`)
to ease the counting of binding votes.

In addition to the vote, voters can also express their justification as part of the comment.
**-1** votes must include justification to allow meaningful discussion.
Any **-1** vote not accompanied by justification is considered invalid.

For votes conducted on GitHub Discussions,
each vote should be cast as an independent comment instead of as a reply within a comment.
This ensures that people can discuss the vote as replies to that specific comment if needed
(e.g., to discuss **-1** vetoes or address concerns).

## Binding Votes

Only votes from the binding voters are counted for each decision,
but other people in the community are also encouraged to cast non-binding votes.
Binding voters should consider any concern from non-binding voters during the vote process.

## Vetoes

A **-1** binding vote is considered a veto for all decision types. Vetoes:

- Stop the proposal until the concerns are resolved
- Cannot be overruled
- Trigger consensus gathering to address concerns

## Voting Requirements

| Decision Type                                                                 | +1 Votes Required                            | Binding Voters                 | Location                              | Minimum Period |
|-------------------------------------------------------------------------------|----------------------------------------------|--------------------------------|---------------------------------------|----------------|
| Governance process and structure modifications                                | 3                                            | PMC                            | Private Mailing List                  | 1 week         |
| Changes in maintainers and PMC rosters                                        | 3 (excluding the people proposed for change) | PMC                            | Private Mailing List                  | 1 week         |
| Incubating subproject graduation to subproject                                | 3                                            | PMC                            | GitHub Discussions                    | 3 days         |
| Subproject management                                                         | 1                                            | PMC                            | GitHub Discussions                    | N/A            |
| Release a new stable major version of the core project                            | 3                                            | PMC                            | GitHub Discussions                    | 3 days         |
| Release a new stable minor version of the core project                            | 3                                            | PMC                            | GitHub Discussions                    | 3 days         |
| Release a new stable patch version of the core project                            | 3                                            | PMC                            | GitHub Discussions                    | N/A            |
| Lance Format Specification modifications                                      | 3 (excluding proposer)                       | PMC                            | GitHub Discussions (with a GitHub PR) | 1 week         |
| Experimental Lance Format Specification feature (stabilization vote)          | 3 (excluding proposer)                       | PMC                            | GitHub Discussions (with a GitHub PR) | 1 week         |
| Code modifications in the core project (except changes to format specifications)  | 1 (excluding proposer)                       | Maintainers with write access  | GitHub PR                             | N/A            |
| Release a new stable version of subprojects                                   | 1                                            | PMC                            | GitHub Discussions                    | N/A            |
| Code modifications in subprojects                                             | 1 (excluding proposer)                       | Contributors with write access | GitHub PR                             | N/A            |

## Experimental Specification Features

Certain format specification changes may be merged as **experimental** before their stabilization vote closes.
This allows iteration on new features without blocking on a completed vote,
while preserving the integrity of the stable format and the community's ability to reject or modify the feature.

### Prerequisites

A feature may only be merged as experimental if it satisfies **all** of the following criteria:

1. The feature is clearly marked as experimental in both the protobuf definitions and the documentation.
2. The feature is **forward compatible**: writers that use the feature do not affect readers that are unaware of it.
3. The feature is **backward compatible**: writers that do not use the feature do not affect readers that use it.
4. Dropping the feature will not require a rewrite of existing data.

### Required Commitments

Before merging an experimental feature, the following commitments must be in place:

1. A Github discussion on the feature has been started.  For features that will span multiple PRs this discussion
should include a design document providing an overview of the entire planned feature.
2. **Authors** accept that if the stabilization vote is rejected or expires without passing, the feature will be removed.
3. **Users** accept that breaking changes may be made to experimental features at any time without a separate vote.
4. **Authors and users** accept that the PMC may request backwards-incompatible changes to the feature during the stabilization process.
5. The file format has an additional concept of "stable versions".  A stable version may not contain any experimental features.  Before a version can be stabilized, all its features must be stabilized or moved out to the next version.

### Stabilization Workflow

1. Open a PR implementing the new format feature.
2. Open a discussion of the feature on GitHub Discussions.  This is not a voting discussion.  It is a place for maintainers
to provide early feedback.
3. Merge the PR with the feature clearly marked as experimental.
4. When ready, open a PR to remove the experimental markers.  This is the PR that will carry the vote.  Merging this PR
stabilized the feature.
5. If the stabilization PR **fails or expires**, remove the feature from the codebase and specification.
