"""Unit tests for the format-spec vote gate logic.

Run with: pytest ci/test_format_vote_gate.py
"""

import pytest

from format_vote_gate import decide_verdict, tally_reviews

HEAD = "sha_head"
PMC = {"alice", "bob", "carol", "dave"}


def is_pmc(login):
    return login is not None and login.lower() in PMC


def review(login, state, commit_id=HEAD):
    return {"login": login, "state": state, "commit_id": commit_id}


def test_counts_distinct_pmc_approvals_on_head_commit():
    approvals, stale, vetoes = tally_reviews(
        [
            review("alice", "APPROVED"),
            review("bob", "APPROVED"),
            review("carol", "APPROVED"),
        ],
        HEAD,
        "author",
        is_pmc,
    )
    assert sorted(approvals) == ["alice", "bob", "carol"]
    assert stale == []
    assert vetoes == []


def test_only_latest_review_per_member_counts():
    # Alice approved, then later requested changes -> she is a veto, not approval.
    approvals, _, vetoes = tally_reviews(
        [review("alice", "APPROVED"), review("alice", "CHANGES_REQUESTED")],
        HEAD,
        "author",
        is_pmc,
    )
    assert approvals == []
    assert vetoes == ["alice"]


def test_approvals_on_earlier_commit_are_stale():
    approvals, stale, _ = tally_reviews(
        [review("alice", "APPROVED", "old_sha"), review("bob", "APPROVED")],
        HEAD,
        "author",
        is_pmc,
    )
    assert approvals == ["bob"]
    assert stale == ["alice"]


def test_ignores_author_non_pmc_and_dismissed():
    approvals, _, vetoes = tally_reviews(
        [
            review("author", "APPROVED"),  # PR author, even if PMC, never counts
            review("eve", "APPROVED"),  # not on the PMC
            review("dave", "DISMISSED"),  # withdrawn
            review("carol", "COMMENTED"),  # a comment is not a vote
        ],
        HEAD,
        "author",
        is_pmc,
    )
    assert approvals == []
    assert vetoes == []


@pytest.mark.parametrize(
    ("veto_count", "approval_count", "period_elapsed", "expected"),
    [
        (1, 5, True, "veto"),  # veto wins even with enough approvals + elapsed
        (0, 2, True, "insufficient"),
        (0, 3, False, "waiting_period"),
        (0, 3, True, "pass"),
    ],
)
def test_decide_verdict_priority(veto_count, approval_count, period_elapsed, expected):
    assert decide_verdict(veto_count, approval_count, period_elapsed, 3) == expected
