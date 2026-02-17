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
| Release a new stable major version of the core project                            | 3                                            | PMC                            | GitHub Discussions                    | 1 week         |
| Release a new stable minor version of the core project                            | 3                                            | PMC                            | GitHub Discussions                    | 3 days         |
| Release a new stable patch version of the core project                            | 3                                            | PMC                            | GitHub Discussions                    | N/A            |
| Lance Format Specification modifications                                      | 3 (excluding proposer)                       | PMC                            | GitHub Discussions (with a GitHub PR) | 1 week         |
| Code modifications in the core project (except changes to format specifications)  | 1 (excluding proposer)                       | Maintainers with write access  | GitHub PR                             | N/A            |
| Release a new stable version of subprojects                                   | 1                                            | PMC                            | GitHub Discussions                    | N/A            |
| Code modifications in subprojects                                             | 1 (excluding proposer)                       | Contributors with write access | GitHub PR                             | N/A            |
