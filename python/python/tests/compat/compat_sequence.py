# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Cross-version index maintenance-sequence search.

Runs on the same per-ref venv substrate as the rest of this package: venv_factory
(venv_manager.py) provisions one venv per ref, so the *setup* half of a sequence runs
under `from_ref` and the *exercise* half under `to_ref` (the version split). After each
run an oracle checks that the reader did not panic and that an index query agrees with a
full (unindexed) scan. This *discovers* cross-version regressions (e.g. ENT-1662)
without hand-coding the triggering sequence.

The scenario is parameterized by index *kind* so every scalar index type gets the same
aged-lifecycle, cross-version treatment. The oracle runs the same predicate twice --
normally and with use_scalar_index=False (lance ignores the index) -- and requires
the results to match. If the two query plans are identical the index wasn't used, so the
comparison is skipped rather than failed (uninformative, not a regression). FTS has no
"ignore the index" mode to diff against, so its oracle reconstructs ground truth from a
full scan: tokenize every live row, then require an FTS search for a spread of sampled
terms to return exactly the rows that contain them. The FTS scenarios run under both
on-disk format versions (LANCE_FTS_FORMAT_VERSION 1 and 2), which take different merge
paths.

The op vocabulary and bounds are deliberately small so the search is runnable; this is
exhaustive over the maintenance-lifecycle grammar up to the configured lengths, not over
every op permutation.
"""

import itertools
import os
import shutil
from pathlib import Path

ROWS_PER_WRITE = 200

SETUP_TAIL_OPS = ["D", "C", "W"]
EXERCISE_OPS = ["W", "D", "C", "Oa", "Om", "Od"]

OP_NAMES = {
    "W": "write rows",
    "I": "create index",
    "D": "delete rows",
    "C": "compact",
    "Oa": "optimize (append)",
    "Om": "optimize (merge)",
    "Od": "optimize",
}


def describe(kind, from_ref, to_ref, setup_ops, exercise_ops, fts_version=None):
    """A plain-English description of a scenario for failure output."""
    writer = ", then ".join(OP_NAMES[o] for o in ["W", "I", *setup_ops])
    reader = ", then ".join(OP_NAMES[o] for o in exercise_ops)
    tag = f" (fts fmt v{fts_version})" if fts_version is not None else ""
    return f"{kind}{tag} ({from_ref} -> {to_ref}): writer [{writer}]; reader [{reader}]"


# Index kinds covered by the maintenance-sequence search.
SCALAR_KINDS = ["BTREE", "BITMAP", "LABEL_LIST", "NGRAM", "ZONEMAP", "BLOOMFILTER"]
ALL_KINDS = ["INVERTED", *SCALAR_KINDS]


class IndexScenario:
    """A picklable, kind-parameterized scenario run across a version split."""

    def __init__(self, kind, path, setup_ops, exercise_ops):
        self.kind = kind
        self.path = str(path)
        self.setup_ops = list(setup_ops)
        self.exercise_ops = list(exercise_ops)
        self.next_idx = 0

    # --- in-venv helpers (only lance + pyarrow available) ---
    def _open(self):
        import lance

        session = lance.Session(index_cache_size_bytes=0, metadata_cache_size_bytes=0)
        return lance.dataset(self.path, session=session)

    def _batch(self, a, b):
        import pyarrow as pa

        idx = list(range(a, b))
        if self.kind == "INVERTED":
            # Each row's text mixes tokens of different frequency: a unique term, a
            # mid-frequency bucket (~1/7 of rows), and one shared by every row. Sampling
            # across that spread exercises postings of varied length.
            return pa.table(
                {"idx": idx, "key": [f"term{i} bucket{i % 7} shared" for i in idx]}
            )
        if self.kind == "LABEL_LIST":
            return pa.table({"idx": idx, "key": [[f"l{i % 8}"] for i in idx]})
        if self.kind == "NGRAM":
            return pa.table({"idx": idx, "key": [f"w{i % 50}x" for i in idx]})
        # BTREE / BITMAP / ZONEMAP / BLOOMFILTER: integer column
        card = 8 if self.kind == "BITMAP" else 50
        key = [i if self.kind == "ZONEMAP" else i % card for i in idx]
        return pa.table({"idx": idx, "key": key})

    def _index_type(self):
        return "INVERTED" if self.kind == "INVERTED" else self.kind

    def _oracle_pred(self):
        if self.kind == "LABEL_LIST":
            return "array_has_any(key, ['l3'])"
        if self.kind == "NGRAM":
            return "contains(key, 'w3x')"
        if self.kind == "ZONEMAP":
            return "key >= 100 AND key < 300"
        return "key == 3"  # BTREE / BITMAP / BLOOMFILTER

    # --- ops ---
    def _op_W(self):
        import lance

        a, b = self.next_idx, self.next_idx + ROWS_PER_WRITE
        self.next_idx = b
        tbl = self._batch(a, b)
        if not os.path.exists(self.path):
            lance.write_dataset(tbl, self.path)  # single fragment
        else:
            self._open().insert(tbl)

    def _op_I(self):
        kwargs = {"with_position": True} if self.kind == "INVERTED" else {}
        self._open().create_scalar_index("key", self._index_type(), **kwargs)

    def _op_D(self):
        # Partial-range delete inside the id space so compaction rewrites and remaps the
        # index per-row.
        if self.next_idx == 0:
            return
        lo, hi = self.next_idx // 4, self.next_idx // 2
        if hi > lo:
            self._open().delete(f"idx >= {lo} AND idx < {hi}")

    def _op_C(self):
        self._open().optimize.compact_files()

    def _op_Oa(self):
        self._open().optimize.optimize_indices(num_indices_to_merge=0)

    def _op_Om(self):
        self._open().optimize.optimize_indices(num_indices_to_merge=10)

    def _op_Od(self):
        self._open().optimize.optimize_indices()

    def _run(self, ops):
        for op in ops:
            getattr(self, f"_op_{op}")()

    # --- methods invoked across the version split ---
    def setup(self):
        shutil.rmtree(self.path, ignore_errors=True)
        self.next_idx = 0
        self._run(["W", "I"] + self.setup_ops)
        return self.next_idx

    def exercise_and_check(self):
        self._run(self.exercise_ops)
        ds = self._open()
        if self.kind == "INVERTED":
            # Differential oracle: rebuild the token -> rows map from a full (unindexed)
            # scan, then require an FTS search for a spread of sampled terms to return
            # exactly those rows. Catches a merge that drops or misassigns postings, not
            # just a row-count drift. (Tokens here are alphanumeric and space-separated,
            # so a whitespace split reproduces lance's tokenization.)
            rows = ds.to_table(columns=["idx", "key"])
            idxs = rows.column("idx").to_pylist()
            texts = rows.column("key").to_pylist()
            truth = {}
            for i, text in zip(idxs, texts):
                for tok in text.split():
                    truth.setdefault(tok, set()).add(i)
            if not truth:
                return  # everything deleted; nothing to search
            vocab = sorted(truth)
            # A spread across the vocabulary plus the most common term.
            sample = set(vocab[:: max(1, len(vocab) // 6)])
            sample.add(max(truth, key=lambda t: len(truth[t])))
            for term in sorted(sample):
                hit = ds.to_table(full_text_query={"query": term, "columns": ["key"]})
                got = set(hit.column("idx").to_pylist())
                want = truth[term]
                assert got == want, (
                    f"FTS('{term}'): index returned {len(got)} rows, corpus has "
                    f"{len(want)} (missing {sorted(want - got)[:5]}, "
                    f"extra {sorted(got - want)[:5]})"
                )
            return
        # Same column/predicate, index on vs forced off: use_scalar_index=False makes
        # lance ignore the index, so the plans differ iff the index is used. If they are
        # identical the index wasn't consulted here (the planner chose a scan after
        # deletes), so the comparison is vacuous -- skip rather than compare two scans.
        pred = self._oracle_pred()
        plan_index = ds.scanner(filter=pred).explain_plan(True)
        plan_scan = ds.scanner(filter=pred, use_scalar_index=False).explain_plan(True)
        if plan_index == plan_scan:
            return
        got = ds.to_table(filter=pred).num_rows
        expected = ds.to_table(filter=pred, use_scalar_index=False).num_rows
        assert got == expected, (
            f"{self.kind}: index gave {got} rows, full scan {expected}, for '{pred}'"
        )


def generate(max_length):
    """Yield every (setup_ops, exercise_ops) whose combined length is 1..max_length,
    breadth-first by total length (shorter first). `max_length` is the number of
    maintenance ops after the implicit write + create-index, split between the writer
    (setup) and reader (exercise) at every position. The order is neutral, so finding a
    bug is a real search, not a sorted shortcut. The space grows fast with max_length,
    so deeper bugs (ENT-1662 needs length 5) cost more to reach."""
    for total in range(1, max_length + 1):
        for setup_len in range(total):  # exercise gets total - setup_len >= 1
            for s in itertools.product(SETUP_TAIL_OPS, repeat=setup_len):
                for e in itertools.product(EXERCISE_OPS, repeat=total - setup_len):
                    yield list(s), list(e)


def search(
    venv_factory,
    from_ref,
    to_ref,
    base_path,
    kind,
    max_length=4,
    shard=0,
    num_shards=1,
    stop_on_first=True,
    fts_version=None,
):
    """Search index-maintenance sequences up to `max_length` ops for one `kind`, across
    (from_ref -> to_ref). Runs only scenarios in this shard (i % num_shards == shard) so
    the space can be split across parallel workers. For INVERTED, `fts_version` ("1" or
    "2") pins the on-disk FTS format (LANCE_FTS_FORMAT_VERSION) on both sides; both are
    Fst token sets and exercise distinct merge paths. Returns failures; stops on the
    first when `stop_on_first`."""
    from_venv = venv_factory.get_venv(from_ref)
    to_venv = venv_factory.get_venv(to_ref)
    env = {}
    if kind == "INVERTED" and fts_version is not None:
        env["LANCE_FTS_FORMAT_VERSION"] = str(fts_version)
    base = Path(base_path)
    failures = []
    # Each setup's aged dataset is built once under from_ref and snapshotted; every
    # exercise for that setup runs on a *copy* of it (a dir copy is far cheaper
    # than rebuilding the index). Cached per shard, keyed by the setup ops.
    snapshots = {}  # tuple(setup) -> (snapshot_path, next_idx), or None if setup failed
    try:
        for i, (setup_tail, exercise) in enumerate(generate(max_length)):
            if i % num_shards != shard:
                continue
            key = tuple(setup_tail)
            if key not in snapshots:
                snap = base / f"snap_{kind}_{len(snapshots)}"
                shutil.rmtree(snap, ignore_errors=True)
                builder = IndexScenario(kind, snap, setup_tail, [])
                try:
                    next_idx = from_venv.execute_method(builder, "setup", env)
                    snapshots[key] = (snap, next_idx)
                except Exception as e:
                    label = describe(
                        kind, from_ref, to_ref, setup_tail, [], fts_version
                    )
                    err = str(e).strip()
                    failures.append({"run": i, "sequence": label, "error": err})
                    snapshots[key] = None
                    shutil.rmtree(snap, ignore_errors=True)
                    if stop_on_first:
                        break
            entry = snapshots[key]
            if entry is None:
                continue  # setup failed; skip its exercises
            snap, next_idx = entry
            ex_path = base / f"ex_{kind}_{i}"
            shutil.rmtree(ex_path, ignore_errors=True)
            shutil.copytree(snap, ex_path)
            scenario = IndexScenario(kind, ex_path, setup_tail, exercise)
            scenario.next_idx = next_idx
            label = describe(kind, from_ref, to_ref, setup_tail, exercise, fts_version)
            try:
                to_venv.execute_method(scenario, "exercise_and_check", env)
            except Exception as e:
                error = str(e).strip()
                failures.append({"run": i, "sequence": label, "error": error})
                if stop_on_first:
                    break
            finally:
                shutil.rmtree(ex_path, ignore_errors=True)
    finally:
        for entry in snapshots.values():
            if entry is not None:
                shutil.rmtree(entry[0], ignore_errors=True)
    return failures
