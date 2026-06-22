"""Format-specification vote gate (see `.github/workflows/format-vote-gate.yml`).

Structurally enforces the PMC vote required for Lance format-specification
changes (https://lance.org/community/voting/). The `format-change` label is
applied by the path labeler (`.github/labeler-area.yml`); this script reads it
and publishes the `format-spec-vote` commit status, which blocks merging until:

  * 3 PMC members have approved the PR (excluding the author), counted only on
    the head commit so new pushes invalidate stale approvals;
  * no PMC member has an outstanding "Request changes" review (a veto); and
  * the 1-week voting period (from when `format-change` was first applied) has
    elapsed.

A PMC member can waive a trivial edit by applying the `format-waived` label.
Non-format PRs get a passing status immediately and are otherwise left alone.

The vote-counting rules are pure functions (`tally_reviews`, `decide_verdict`)
unit tested in `test_format_vote_gate.py`; `main` wires them to the GitHub API.
"""

import json
import os
from datetime import datetime, timedelta, timezone

STATUS_CONTEXT = "format-spec-vote"
FORMAT_LABEL = "format-change"
WAIVED_LABEL = "format-waived"
COMMENT_MARKER = "<!-- format-spec-vote-status -->"
REQUIRED_APPROVALS = 3
PERIOD_DAYS = 7
VOTING_URL = "https://lance.org/community/voting/"

# Review states that express a stance; COMMENTED/PENDING are ignored.
_STANCE_STATES = ("APPROVED", "CHANGES_REQUESTED", "DISMISSED")


def tally_reviews(reviews, head_sha, author, is_pmc):
    """Tally PMC votes from a PR's reviews.

    `reviews` is an ordered list of dicts with `login`, `state`, `commit_id`.
    A member's stance is their most recent stance review. Approvals only count
    on the head commit; earlier ones are stale. A "changes requested" review is
    a veto regardless of commit. The PR author never counts.
    """
    latest = {}
    for review in reviews:
        login = review["login"]
        if not login or not is_pmc(login) or login == author:
            continue
        if review["state"] not in _STANCE_STATES:
            continue
        latest[login.lower()] = review

    approvals, stale_approvals, vetoes = [], [], []
    for review in latest.values():
        if review["state"] == "APPROVED":
            target = approvals if review["commit_id"] == head_sha else stale_approvals
            target.append(review["login"])
        elif review["state"] == "CHANGES_REQUESTED":
            vetoes.append(review["login"])
    return approvals, stale_approvals, vetoes


def decide_verdict(veto_count, approval_count, period_elapsed, required):
    """Return the blocking condition (if any), in priority order."""
    if veto_count > 0:
        return "veto"
    if approval_count < required:
        return "insufficient"
    if not period_elapsed:
        return "waiting_period"
    return "pass"


def _fmt_list(logins):
    return ", ".join(f"@{login}" for login in logins) if logins else "none"


def _as_utc(dt):
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _build_comment(headline, approval_cell, vetoes, period_cell):
    return "\n".join(
        [
            COMMENT_MARKER,
            "> [!IMPORTANT]",
            "> ## Format specification vote",
            "",
            "This PR modifies the Lance format specification, so it requires "
            f"**{REQUIRED_APPROVALS} binding +1 votes from PMC members** "
            "(excluding the proposer) and a minimum "
            f"**{PERIOD_DAYS}-day** voting period before it can merge. "
            "Vote by approving this PR (+1) or requesting changes (−1, a veto). "
            f"See the [voting process]({VOTING_URL}).",
            "",
            f"**Status: {headline}**",
            "",
            "| | |",
            "|---|---|",
            f"| Approvals (this commit) | {approval_cell} |",
            f"| Vetoes | {_fmt_list(vetoes)} |",
            f"| Voting period | {period_cell} |",
            "",
            "<sub>Updated automatically by the format-spec vote gate. A PMC member "
            f"may apply the `{WAIVED_LABEL}` label to waive the vote for a trivial "
            "edit (typo, wording, formatting).</sub>",
        ]
    )


def _load_pmc(workspace):
    import yaml

    roster_path = os.path.join(workspace, "docs", "src", "community", "pmc.yaml")
    with open(roster_path) as handle:
        roster = yaml.safe_load(handle)
    return {member["handle"].lower() for member in roster["members"]}


class Gate:
    def __init__(self, repo, pmc, run_url):
        self.repo = repo
        self.pmc = pmc
        self.run_url = run_url

    def is_pmc(self, login):
        return login is not None and login.lower() in self.pmc

    def set_status(self, sha, state, description):
        self.repo.get_commit(sha).create_status(
            state=state,
            context=STATUS_CONTEXT,
            description=description[:140],
            target_url=self.run_url,
        )

    def upsert_comment(self, issue, body):
        for comment in issue.get_comments():
            if COMMENT_MARKER in (comment.body or ""):
                if comment.body != body:
                    comment.edit(body)
                return
        issue.create_comment(body)

    def label_facts(self, issue):
        """When `format-change` was first applied, and whether a PMC waived."""
        first_added = None
        waived_by_pmc = False
        for event in issue.get_events():
            if event.event not in ("labeled", "unlabeled") or event.label is None:
                continue
            name = event.label.name
            actor = event.actor.login if event.actor else None
            if (
                name == FORMAT_LABEL
                and event.event == "labeled"
                and first_added is None
            ):
                first_added = _as_utc(event.created_at)
            elif (
                name == WAIVED_LABEL and event.event == "labeled" and self.is_pmc(actor)
            ):
                waived_by_pmc = True
        return first_added, waived_by_pmc

    def evaluate(self, number):
        pr = self.repo.get_pull(number)
        if pr.state != "open":
            print(f"PR #{number} is {pr.state}; skipping.")
            return
        head_sha = pr.head.sha
        labels = {label.name for label in pr.labels}

        # Non-format PRs get a passing status and are otherwise left alone.
        if FORMAT_LABEL not in labels:
            self.set_status(
                head_sha, "success", "No format-spec change; vote not required."
            )
            print(f"PR #{number}: not a format change.")
            return

        issue = self.repo.get_issue(number)
        first_added, waived = self.label_facts(issue)

        if WAIVED_LABEL in labels and waived:
            self.set_status(
                head_sha, "success", "Format-spec vote waived by a PMC member."
            )
            print(f"PR #{number}: vote waived.")
            return

        reviews = [
            {
                "login": review.user.login if review.user else None,
                "state": review.state,
                "commit_id": review.commit_id,
            }
            for review in pr.get_reviews()
        ]
        approvals, stale, vetoes = tally_reviews(
            reviews, head_sha, pr.user.login, self.is_pmc
        )

        now = datetime.now(timezone.utc)
        vote_start = first_added or now
        period_ends = vote_start + timedelta(days=PERIOD_DAYS)
        period_elapsed = now >= period_ends
        verdict = decide_verdict(
            len(vetoes), len(approvals), period_elapsed, REQUIRED_APPROVALS
        )

        end_date = period_ends.date().isoformat()
        if verdict == "veto":
            state, summary = "failure", f"Vetoed by {len(vetoes)} PMC member(s)."
            headline = f"❌ Blocked — vetoed by {_fmt_list(vetoes)}"
        elif verdict == "insufficient":
            state = "failure"
            summary = (
                f"{len(approvals)}/{REQUIRED_APPROVALS} PMC approvals on this commit."
            )
            headline = f"❌ Blocked — {len(approvals)} of {REQUIRED_APPROVALS} required approvals"
        elif verdict == "waiting_period":
            state, summary = "failure", f"Approved; voting period ends {end_date}."
            headline = (
                f"⏳ Approvals met ({len(approvals)}/{REQUIRED_APPROVALS}); "
                f"voting period ends {end_date}"
            )
        else:
            state = "success"
            summary = f"Passed — {len(approvals)} PMC approvals, period elapsed."
            headline = f"✅ Vote passed — {len(approvals)} PMC approvals, voting period elapsed"

        days_left = max(0, -(-(period_ends - now).days))  # ceil of day difference
        period_cell = (
            f"elapsed (ended {end_date})"
            if period_elapsed
            else f"ends {end_date} ({days_left} day(s) left)"
        )
        approval_cell = (
            f"{_fmt_list(approvals)} ({len(approvals)}/{REQUIRED_APPROVALS})"
        )
        if stale:
            approval_cell += f" — stale, re-approve needed: {_fmt_list(stale)}"

        self.set_status(head_sha, state, summary)
        self.upsert_comment(
            issue, _build_comment(headline, approval_cell, vetoes, period_cell)
        )
        print(f"PR #{number}: {summary}")


def main():
    from github import Github

    workspace = os.environ["GITHUB_WORKSPACE"]
    token = os.environ["GITHUB_TOKEN"]
    repo_name = os.environ["GITHUB_REPOSITORY"]
    event_name = os.environ["GITHUB_EVENT_NAME"]
    run_url = (
        f"{os.environ['GITHUB_SERVER_URL']}/{repo_name}/actions/runs/"
        f"{os.environ['GITHUB_RUN_ID']}"
    )

    repo = Github(token).get_repo(repo_name)
    gate = Gate(repo, _load_pmc(workspace), run_url)

    if event_name == "schedule":
        # The schedule trigger has no PR context, so sweep every open
        # format-change PR to re-check the voting-period clock.
        pulls = [
            pr
            for pr in repo.get_pulls(state="open")
            if any(label.name == FORMAT_LABEL for label in pr.labels)
        ]
        print(f"Scheduled sweep: {len(pulls)} open {FORMAT_LABEL} PR(s).")
        for pr in pulls:
            try:
                gate.evaluate(pr.number)
            except Exception as err:  # noqa: BLE001 - keep sweeping other PRs
                print(f"PR #{pr.number}: {err}")
    else:
        with open(os.environ["GITHUB_EVENT_PATH"]) as handle:
            event = json.load(handle)
        gate.evaluate(event["pull_request"]["number"])


if __name__ == "__main__":
    main()
