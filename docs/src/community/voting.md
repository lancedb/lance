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
| Lance Format Specification modifications                                      | 3 (excluding proposer)                       | PMC                            | GitHub PR (see [below](#lance-format-specification-vote-gate)) | 1 week         |
| Code modifications in the core project (except changes to format specifications)  | 1 (excluding proposer)                       | Maintainers with write access  | GitHub PR                             | N/A            |
| Release a new stable version of subprojects                                   | 1                                            | PMC                            | GitHub Discussions                    | N/A            |
| Code modifications in subprojects                                             | 1 (excluding proposer)                       | Contributors with write access | GitHub PR                             | N/A            |

## Lance Format Specification Vote Gate

Votes on Lance format-specification changes are cast and counted directly on the
pull request, and the requirement is enforced structurally in CI rather than by
convention.

A PR is considered a format-specification change when it modifies the protobuf
definitions (`protos/**/*.proto`) or the spec documentation (`docs/src/format/**`);
such PRs are labelled `format-change` automatically. The
[format spec vote gate](https://github.com/lance-format/lance/blob/main/.github/workflows/format-vote-gate.yml)
blocks merging a `format-change` PR until all of the following hold:

- **Three binding +1 votes.** Three PMC members have approved the PR, excluding
  the proposer. Cast +1 by approving the PR. Only approvals on the latest commit
  count — pushing new commits invalidates earlier approvals, since the proposal
  has changed.
- **No veto.** No PMC member has an outstanding "Request changes" review. A `-1`
  binding vote (cast by requesting changes) is a veto and blocks the merge until
  withdrawn.
- **Minimum voting period.** At least one week has elapsed since the PR was
  flagged as a format change.

The gate is the `format-spec-vote` required status check on protected branches.
The PMC roster used to count votes is read from
[`docs/src/community/pmc.yaml`](./pmc.md).

For a trivial edit that does not change the format — a typo, wording, or
formatting fix — a PMC member may apply the `format-waived` label to waive the
vote.
