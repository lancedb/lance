# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

# Benchmarks for Full Text Search (FTS) queries on Wikipedia dataset.
#
# Tests various query types (basic, match, phrase) with different
# parameters (K values, cache settings) to measure FTS latency.
#
# This benchmark is loosely modeld after the Quickwit benchmark located
# at https://github.com/quickwit-oss/search-benchmark-game and uses a
# similar Wikipedia dataset.  However, the dataset used by this benchmark
# comes from HuggingFace and is smaller so it can't be compared directly.

import lance
import pytest
from ci_benchmarks.datasets import get_dataset_uri
from ci_benchmarks.utils import wipe_os_cache

# K values for result limits
K_VALUES = [10, 100, 1000]
K_LABELS = ["k10", "k100", "k1000"]

# Test queries - common Wikipedia search terms
BASIC_QUERIES = [
    "lost episode",
    "artificial intelligence",
    "database systems",
]

BASIC_QUERY_LABELS = [
    "lost_episode",
    "artificial_intelligence",
    "database_systems",
]

# Phrase queries for exact matching
PHRASE_QUERIES = [
    '"machine learning algorithm"',
    '"artificial intelligence research"',
]

PHRASE_QUERY_LABELS = [
    "phrase_machine_learning_algorithm",
    "phrase_artificial_intelligence_research",
]

ALL_QUERIES = BASIC_QUERIES + PHRASE_QUERIES
ALL_QUERY_LABELS = BASIC_QUERY_LABELS + PHRASE_QUERY_LABELS


@pytest.mark.parametrize("k", K_VALUES, ids=K_LABELS)
@pytest.mark.parametrize("query", ALL_QUERIES, ids=ALL_QUERY_LABELS)
@pytest.mark.parametrize("use_cache", [True, False], ids=["cache", "no_cache"])
def test_query(benchmark, k, query, use_cache):
    """Benchmark basic FTS string query."""
    dataset_uri = get_dataset_uri("wikipedia")
    ds = lance.dataset(dataset_uri)

    def clear_cache():
        wipe_os_cache(dataset_uri)

    def bench():
        to_search = ds if use_cache else lance.dataset(dataset_uri)
        to_search.to_table(full_text_query=query, limit=k, columns=["_rowid"])

    setup = None if use_cache else clear_cache
    warmup_rounds = 1 if use_cache else 0

    benchmark.pedantic(
        bench,
        warmup_rounds=warmup_rounds,
        rounds=100,
        iterations=1,
        setup=setup,
    )
