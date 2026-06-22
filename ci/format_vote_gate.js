// Logic for the format-specification vote gate
// (`.github/workflows/format-vote-gate.yml`). Extracted into a module so the
// vote-counting rules can be unit tested (see `format_vote_gate.test.js`); the
// workflow loads it from the trusted base checkout and calls `run`.

const fs = require("fs");
const path = require("path");

const STATUS_CONTEXT = "format-spec-vote";
const LABEL = "format-change";
const COMMENT_MARKER = "<!-- format-spec-vote-status -->";
const REQUIRED_APPROVALS = 3;
const PERIOD_DAYS = 7;
const PERIOD_MS = PERIOD_DAYS * 24 * 60 * 60 * 1000;
const VOTING_URL = "https://lance.org/community/voting/";
const DAY_MS = 24 * 60 * 60 * 1000;

// The Lance format specification is the proto definitions and the spec docs.
// Changes to either require a PMC vote.
function isFormatFile(p) {
  return (
    (p.endsWith(".proto") && p.startsWith("protos/")) ||
    p.startsWith("docs/src/format/")
  );
}

// Each PMC member's current stance is their most recent non-comment review.
// Approvals are only counted on the head commit (stale approvals from earlier
// commits do not count toward the bar); a "changes requested" review is a veto
// and blocks regardless of commit until withdrawn. The PR author never counts,
// even if they are on the PMC.
function tallyReviews(reviews, headSha, author, isPmc) {
  const latest = new Map();
  for (const r of reviews) {
    const login = r.user && r.user.login;
    if (!login || !isPmc(login) || login === author) continue;
    if (!["APPROVED", "CHANGES_REQUESTED", "DISMISSED"].includes(r.state)) {
      continue;
    }
    latest.set(login.toLowerCase(), {
      login,
      state: r.state,
      commitId: r.commit_id,
    });
  }
  const approvals = [];
  const staleApprovals = [];
  const vetoes = [];
  for (const v of latest.values()) {
    if (v.state === "APPROVED") {
      (v.commitId === headSha ? approvals : staleApprovals).push(v.login);
    } else if (v.state === "CHANGES_REQUESTED") {
      vetoes.push(v.login);
    }
  }
  return { approvals, staleApprovals, vetoes };
}

// Pure decision: which blocking condition (if any) applies, in priority order.
function decideVerdict({ vetoCount, approvalCount, periodElapsed, required }) {
  if (vetoCount > 0) return "veto";
  if (approvalCount < required) return "insufficient";
  if (!periodElapsed) return "waiting_period";
  return "pass";
}

const fmtList = (xs) => (xs.length ? xs.map((u) => `@${u}`).join(", ") : "none");

function loadPmc(workspace) {
  const yaml = require(path.join(workspace, "node_modules", "js-yaml"));
  const rosterPath = path.join(workspace, "docs", "src", "community", "pmc.yaml");
  const roster = yaml.load(fs.readFileSync(rosterPath, "utf8"));
  return new Set(roster.members.map((m) => m.handle.toLowerCase()));
}

async function run({ github, context, core }) {
  const { owner, repo } = context.repo;
  const workspace = process.env.GITHUB_WORKSPACE;
  const pmc = loadPmc(workspace);
  const isPmc = (login) => login != null && pmc.has(login.toLowerCase());
  const runUrl = `${context.serverUrl}/${owner}/${repo}/actions/runs/${context.runId}`;

  async function setStatus(sha, state, description) {
    await github.rest.repos.createCommitStatus({
      owner,
      repo,
      sha,
      state,
      context: STATUS_CONTEXT,
      description: description.slice(0, 140),
      target_url: runUrl,
    });
  }

  async function upsertComment(prNumber, body) {
    const full = `${COMMENT_MARKER}\n${body}`;
    const comments = await github.paginate(github.rest.issues.listComments, {
      owner,
      repo,
      issue_number: prNumber,
      per_page: 100,
    });
    const existing = comments.find((c) => c.body && c.body.includes(COMMENT_MARKER));
    if (existing) {
      if (existing.body !== full) {
        await github.rest.issues.updateComment({
          owner,
          repo,
          comment_id: existing.id,
          body: full,
        });
      }
    } else {
      await github.rest.issues.createComment({
        owner,
        repo,
        issue_number: prNumber,
        body: full,
      });
    }
  }

  // Returns { firstAddedAt, removedByPmc } for LABEL on this PR, from the
  // timeline: when the label was first applied (the vote-period anchor) and
  // whether a PMC member deliberately removed it (a trivial-edit waiver).
  async function labelHistory(prNumber) {
    const events = await github.paginate(
      github.rest.issues.listEventsForTimeline,
      { owner, repo, issue_number: prNumber, per_page: 100 },
    );
    let firstAddedAt = null;
    let removedByPmc = false;
    for (const e of events) {
      if (!e.label || e.label.name !== LABEL) continue;
      if (e.event === "labeled" && !firstAddedAt) {
        firstAddedAt = e.created_at;
      } else if (e.event === "unlabeled" && isPmc(e.actor && e.actor.login)) {
        removedByPmc = true;
      }
    }
    return { firstAddedAt, removedByPmc };
  }

  async function evaluate(prNumber) {
    const { data: pr } = await github.rest.pulls.get({
      owner,
      repo,
      pull_number: prNumber,
    });
    if (pr.state !== "open") {
      core.info(`PR #${prNumber} is ${pr.state}; skipping.`);
      return;
    }
    const headSha = pr.head.sha;
    const author = pr.user.login;

    const files = await github.paginate(github.rest.pulls.listFiles, {
      owner,
      repo,
      pull_number: prNumber,
      per_page: 100,
    });
    const touchesFormat = files.map((f) => f.filename).some(isFormatFile);

    let hasLabel = pr.labels.some((l) => l.name === LABEL);
    const { firstAddedAt, removedByPmc } = await labelHistory(prNumber);

    // Auto-apply the label, unless a PMC member deliberately removed it.
    if (touchesFormat && !hasLabel && !removedByPmc) {
      await github.rest.issues.addLabels({
        owner,
        repo,
        issue_number: prNumber,
        labels: [LABEL],
      });
      hasLabel = true;
    }

    if (!hasLabel) {
      await setStatus(
        headSha,
        "success",
        removedByPmc
          ? "Format-spec vote waived by a PMC member."
          : "No format-spec change; vote not required.",
      );
      if (removedByPmc) {
        await upsertComment(
          prNumber,
          `> [!NOTE]\n> A PMC member removed the \`${LABEL}\` label, ` +
            `waiving the format-spec vote for this PR (trivial edit).`,
        );
      }
      core.info(`PR #${prNumber}: no vote required.`);
      return;
    }

    const reviews = await github.paginate(github.rest.pulls.listReviews, {
      owner,
      repo,
      pull_number: prNumber,
      per_page: 100,
    });
    const { approvals, staleApprovals, vetoes } = tallyReviews(
      reviews,
      headSha,
      author,
      isPmc,
    );

    const now = new Date();
    const voteStart = firstAddedAt ? new Date(firstAddedAt) : now;
    const periodEnds = new Date(voteStart.getTime() + PERIOD_MS);
    const periodElapsed = now >= periodEnds;
    const verdict = decideVerdict({
      vetoCount: vetoes.length,
      approvalCount: approvals.length,
      periodElapsed,
      required: REQUIRED_APPROVALS,
    });

    const endDate = periodEnds.toISOString().slice(0, 10);
    let state, summary, headline;
    if (verdict === "veto") {
      state = "failure";
      summary = `Vetoed by ${vetoes.length} PMC member(s).`;
      headline = `❌ Blocked — vetoed by ${fmtList(vetoes)}`;
    } else if (verdict === "insufficient") {
      state = "failure";
      summary = `${approvals.length}/${REQUIRED_APPROVALS} PMC approvals on this commit.`;
      headline = `❌ Blocked — ${approvals.length} of ${REQUIRED_APPROVALS} required approvals`;
    } else if (verdict === "waiting_period") {
      state = "failure";
      summary = `Approved; voting period ends ${endDate}.`;
      headline =
        `⏳ Approvals met (${approvals.length}/${REQUIRED_APPROVALS}); ` +
        `voting period ends ${endDate}`;
    } else {
      state = "success";
      summary = `Passed — ${approvals.length} PMC approvals, period elapsed.`;
      headline = `✅ Vote passed — ${approvals.length} PMC approvals, voting period elapsed`;
    }

    const daysLeft = Math.max(0, Math.ceil((periodEnds.getTime() - now.getTime()) / DAY_MS));
    const periodCell = periodElapsed
      ? `elapsed (ended ${endDate})`
      : `ends ${endDate} (${daysLeft} day(s) left)`;
    const approvalCell =
      `${fmtList(approvals)} (${approvals.length}/${REQUIRED_APPROVALS})` +
      (staleApprovals.length
        ? ` — stale, re-approve needed: ${fmtList(staleApprovals)}`
        : "");

    const body = [
      "> [!IMPORTANT]",
      "> ## Format specification vote",
      "",
      "This PR modifies the Lance format specification, so it requires " +
        `**${REQUIRED_APPROVALS} binding +1 votes from PMC members** ` +
        "(excluding the proposer) and a minimum " +
        `**${PERIOD_DAYS}-day** voting period before it can merge. ` +
        "Vote by approving this PR (+1) or requesting changes (−1, a veto). " +
        `See the [voting process](${VOTING_URL}).`,
      "",
      `**Status: ${headline}**`,
      "",
      "| | |",
      "|---|---|",
      `| Approvals (this commit) | ${approvalCell} |`,
      `| Vetoes | ${fmtList(vetoes)} |`,
      `| Voting period | ${periodCell} |`,
      "",
      "<sub>Updated automatically by the format-spec vote gate. A PMC member " +
        `may remove the \`${LABEL}\` label to waive the vote for a trivial edit ` +
        "(typo, wording, formatting).</sub>",
    ].join("\n");

    await setStatus(headSha, state, summary);
    await upsertComment(prNumber, body);
    core.info(`PR #${prNumber}: ${summary}`);
  }

  if (context.eventName === "schedule") {
    const issues = await github.paginate(github.rest.issues.listForRepo, {
      owner,
      repo,
      state: "open",
      labels: LABEL,
      per_page: 100,
    });
    const pulls = issues.filter((i) => i.pull_request);
    core.info(`Scheduled sweep: ${pulls.length} open ${LABEL} PR(s).`);
    for (const p of pulls) {
      try {
        await evaluate(p.number);
      } catch (err) {
        core.warning(`PR #${p.number}: ${err.message}`);
      }
    }
  } else {
    await evaluate(context.payload.pull_request.number);
  }
}

module.exports = {
  run,
  isFormatFile,
  tallyReviews,
  decideVerdict,
  constants: { STATUS_CONTEXT, LABEL, REQUIRED_APPROVALS, PERIOD_DAYS },
};
