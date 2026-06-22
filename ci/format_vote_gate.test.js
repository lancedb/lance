// Unit tests for the format-spec vote gate logic.
// Run with: node --test ci/format_vote_gate.test.js

const { test } = require("node:test");
const assert = require("node:assert/strict");

const { isFormatFile, tallyReviews, decideVerdict } = require("./format_vote_gate.js");

test("isFormatFile matches protos and spec docs only", () => {
  assert.ok(isFormatFile("protos/table.proto"));
  assert.ok(isFormatFile("docs/src/format/file.md"));
  // .proto outside protos/ (e.g. vendored or test fixtures) is not the spec.
  assert.ok(!isFormatFile("rust/lance-encoding/foo.proto"));
  assert.ok(!isFormatFile("docs/src/community/voting.md"));
  assert.ok(!isFormatFile("README.md"));
  assert.ok(!isFormatFile("protos/AGENTS.md"));
});

const HEAD = "sha_head";
const isPmc = (login) => ["alice", "bob", "carol", "dave"].includes(login.toLowerCase());
const review = (login, state, commitId = HEAD) => ({ user: { login }, state, commit_id: commitId });

test("counts distinct PMC approvals on the head commit", () => {
  const { approvals, vetoes, staleApprovals } = tallyReviews(
    [review("alice", "APPROVED"), review("bob", "APPROVED"), review("carol", "APPROVED")],
    HEAD,
    "author",
    isPmc,
  );
  assert.deepEqual(approvals.sort(), ["alice", "bob", "carol"]);
  assert.equal(vetoes.length, 0);
  assert.equal(staleApprovals.length, 0);
});

test("only a reviewer's latest review counts", () => {
  // Alice approved, then later requested changes -> she is a veto, not approval.
  const { approvals, vetoes } = tallyReviews(
    [review("alice", "APPROVED"), review("alice", "CHANGES_REQUESTED")],
    HEAD,
    "author",
    isPmc,
  );
  assert.deepEqual(approvals, []);
  assert.deepEqual(vetoes, ["alice"]);
});

test("approvals on an earlier commit are stale, not counted", () => {
  const { approvals, staleApprovals } = tallyReviews(
    [review("alice", "APPROVED", "old_sha"), review("bob", "APPROVED")],
    HEAD,
    "author",
    isPmc,
  );
  assert.deepEqual(approvals, ["bob"]);
  assert.deepEqual(staleApprovals, ["alice"]);
});

test("ignores the author, non-PMC reviewers, and dismissed reviews", () => {
  const { approvals, vetoes } = tallyReviews(
    [
      review("author", "APPROVED"), // PR author, even if PMC, never counts
      review("eve", "APPROVED"), // not on the PMC
      review("dave", "DISMISSED"), // withdrawn
      review("carol", "COMMENTED"), // a comment is not a vote
    ],
    HEAD,
    "author",
    isPmc,
  );
  assert.deepEqual(approvals, []);
  assert.deepEqual(vetoes, []);
});

test("decideVerdict applies blocking conditions in priority order", () => {
  // Veto wins even with enough approvals and elapsed period.
  assert.equal(
    decideVerdict({ vetoCount: 1, approvalCount: 5, periodElapsed: true, required: 3 }),
    "veto",
  );
  assert.equal(
    decideVerdict({ vetoCount: 0, approvalCount: 2, periodElapsed: true, required: 3 }),
    "insufficient",
  );
  assert.equal(
    decideVerdict({ vetoCount: 0, approvalCount: 3, periodElapsed: false, required: 3 }),
    "waiting_period",
  );
  assert.equal(
    decideVerdict({ vetoCount: 0, approvalCount: 3, periodElapsed: true, required: 3 }),
    "pass",
  );
});
