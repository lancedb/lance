# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Cross-version index maintenance-sequence search, wired as a compat test.

For each index kind, generate maintenance-op sequences from a small grammar, run the
setup half under an older writer ref and the exercise half under a newer reader, and
check after each that the reader did not panic and that an index query matches a full
scan. Discovers cross-version regressions (e.g. ENT-1662, under INVERTED) with no
hand-coded sequence.

Refs and max length are environment-driven so the suite can run between two refs
(versions, commits, or branches): COMPAT_FROM_REF / COMPAT_TO_REF / COMPAT_MAX_LENGTH /
COMPAT_KINDS (comma-separated subset of kinds) / COMPAT_SHARDS (split each kind's search
into this many cases so pytest-xdist (`-n auto`) parallelizes them across cores).
"""

import os

import pytest

from .compat_decorator import pylance_stable_versions
from .compat_sequence import ALL_KINDS, search


def _default_refs():
    """The two most recent published stable releases (older -> newer)."""
    versions = pylance_stable_versions()
    if len(versions) >= 2:
        return str(versions[-2]), str(versions[-1])
    return "6.0.1", "7.0.0"  # fallback if PyPI is unreachable


_default_from, _default_to = _default_refs()
FROM_REF = os.environ.get("COMPAT_FROM_REF") or _default_from
TO_REF = os.environ.get("COMPAT_TO_REF") or _default_to
MAX_LENGTH = int(os.environ.get("COMPAT_MAX_LENGTH", "4"))
KINDS = os.environ.get("COMPAT_KINDS", ",".join(ALL_KINDS)).split(",")
# Many small shards (default 4x cores) so xdist's dynamic scheduler keeps every worker
# busy and an oversubscribed `-n` has work to overlap.
NUM_SHARDS = int(os.environ.get("COMPAT_SHARDS", str((os.cpu_count() or 1) * 4)))


def _cases():
    """(kind, fts_version) cases. FTS runs under both on-disk formats (v1, v2); the
    scalar kinds are format-agnostic and run once."""
    cases = []
    for kind in KINDS:
        if kind == "INVERTED":
            cases.extend([("INVERTED", "1"), ("INVERTED", "2")])
        else:
            cases.append((kind, None))
    return cases


CASES = _cases()
CASE_IDS = [k if v is None else f"{k}-fmtv{v}" for k, v in CASES]


@pytest.mark.compat
@pytest.mark.parametrize("kind,fts_version", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("shard", range(NUM_SHARDS))
def test_index_maintenance_sequence_search(
    venv_factory, tmp_path, kind, fts_version, shard
):
    failures = search(
        venv_factory,
        FROM_REF,
        TO_REF,
        tmp_path,
        kind,
        max_length=MAX_LENGTH,
        shard=shard,
        num_shards=NUM_SHARDS,
        fts_version=fts_version,
    )
    # First line is the failure itself so it shows in pytest's bottom summary; the rest
    # (if more than one) appears in the failure body.
    assert not failures, "\n".join(
        f"{f['sequence']}  ==>  {f['error']}" for f in failures
    )
