#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Generate a synthetic Lance dataset for FTS benchmarks.

Example:
  python scripts/fts_gen_data.py \
    --uri /tmp/fts_ds \
    --rows 5_000_000 \
    --vocab-size 200_000 \
    --min-tokens 5 \
    --max-tokens 20 \
    --zipf-s 1.1 \
    --batch-size 10000 \
    --emit-terms /tmp/fts_terms.txt \
    --build-index \
    --with-position false
"""

from __future__ import annotations

import argparse
import bisect
import os
import random
from typing import Callable, Iterable, List, Optional

import pyarrow as pa
import lance


def _build_vocab(vocab_size: int) -> List[str]:
    return [f"term_{i:08d}" for i in range(vocab_size)]


def _make_sampler(vocab_size: int, zipf_s: float, seed: int) -> Callable[[random.Random], int]:
    if zipf_s <= 0:
        def _sample_uniform(rng: random.Random) -> int:
            return rng.randrange(vocab_size)
        return _sample_uniform

    weights = [1.0 / ((i + 1) ** zipf_s) for i in range(vocab_size)]
    total = sum(weights)
    cdf: List[float] = []
    acc = 0.0
    for w in weights:
        acc += w / total
        cdf.append(acc)

    def _sample_zipf(rng: random.Random) -> int:
        r = rng.random()
        return bisect.bisect_left(cdf, r)

    return _sample_zipf


def _record_batches(
    rows: int,
    batch_size: int,
    vocab: List[str],
    sampler: Callable[[random.Random], int],
    min_tokens: int,
    max_tokens: int,
    seed: int,
    id_column: str,
    text_column: str,
) -> Iterable[pa.RecordBatch]:
    rng = random.Random(seed)
    for start in range(0, rows, batch_size):
        n = min(batch_size, rows - start)
        ids = list(range(start, start + n))
        texts: List[str] = []
        for _ in range(n):
            token_count = rng.randint(min_tokens, max_tokens)
            tokens = [vocab[sampler(rng)] for _ in range(token_count)]
            texts.append(" ".join(tokens))
        yield pa.record_batch(
            {
                id_column: pa.array(ids, type=pa.int64()),
                text_column: pa.array(texts, type=pa.string()),
            }
        )


def _set_env_if_provided(name: str, value: Optional[int]) -> None:
    if value is not None:
        os.environ[name] = str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FTS benchmark dataset")
    parser.add_argument("--uri", required=True, help="Dataset URI")
    parser.add_argument("--rows", type=int, default=1_000_000, help="Number of rows")
    parser.add_argument("--vocab-size", type=int, default=100_000, help="Vocabulary size")
    parser.add_argument("--min-tokens", type=int, default=5, help="Min tokens per doc")
    parser.add_argument("--max-tokens", type=int, default=20, help="Max tokens per doc")
    parser.add_argument(
        "--zipf-s",
        type=float,
        default=0.0,
        help="Zipf s for term popularity (0 = uniform)",
    )
    parser.add_argument("--batch-size", type=int, default=10_000, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--id-column",
        default="id",
        help="Name of int64 id column",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Name of text column",
    )
    parser.add_argument(
        "--mode",
        default="overwrite",
        choices=["create", "overwrite", "append"],
        help="Write mode",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=1_000_000,
        help="Max rows per data file",
    )
    parser.add_argument(
        "--emit-terms",
        default=None,
        help="Write vocab terms to this file (one per line)",
    )

    # Index options
    parser.add_argument("--build-index", action="store_true", help="Build FTS index")
    parser.add_argument("--with-position", action="store_true", help="Store positions")
    parser.add_argument(
        "--base-tokenizer",
        default="simple",
        help="Base tokenizer for INVERTED index",
    )
    parser.add_argument("--language", default="English", help="Tokenizer language")
    parser.add_argument(
        "--max-token-length",
        type=int,
        default=40,
        help="Max token length",
    )
    parser.add_argument("--lower-case", action="store_true", help="Lower-case tokens")
    parser.add_argument("--stem", action="store_true", help="Apply stemming")
    parser.add_argument(
        "--remove-stop-words",
        action="store_true",
        help="Remove stop words",
    )
    parser.add_argument(
        "--custom-stop-words",
        default=None,
        help="Comma-separated custom stop words",
    )
    parser.add_argument(
        "--ascii-folding",
        action="store_true",
        help="ASCII folding",
    )
    parser.add_argument(
        "--min-ngram-length",
        type=int,
        default=None,
        help="Min ngram length",
    )
    parser.add_argument(
        "--max-ngram-length",
        type=int,
        default=None,
        help="Max ngram length",
    )
    parser.add_argument(
        "--prefix-only",
        action="store_true",
        help="Use prefix-only ngram",
    )

    # FTS partitioning env overrides
    parser.add_argument(
        "--fts-partition-size-mb",
        type=int,
        default=None,
        help="Set LANCE_FTS_PARTITION_SIZE",
    )
    parser.add_argument(
        "--fts-target-size-mb",
        type=int,
        default=None,
        help="Set LANCE_FTS_TARGET_SIZE",
    )
    parser.add_argument(
        "--fts-num-shards",
        type=int,
        default=None,
        help="Set LANCE_FTS_NUM_SHARDS",
    )
    parser.add_argument(
        "--fts-flush-size-mb",
        type=int,
        default=None,
        help="Set LANCE_FTS_FLUSH_SIZE",
    )

    args = parser.parse_args()

    _set_env_if_provided("LANCE_FTS_PARTITION_SIZE", args.fts_partition_size_mb)
    _set_env_if_provided("LANCE_FTS_TARGET_SIZE", args.fts_target_size_mb)
    _set_env_if_provided("LANCE_FTS_NUM_SHARDS", args.fts_num_shards)
    _set_env_if_provided("LANCE_FTS_FLUSH_SIZE", args.fts_flush_size_mb)

    vocab = _build_vocab(args.vocab_size)
    if args.emit_terms:
        with open(args.emit_terms, "w", encoding="utf-8") as f:
            for term in vocab:
                f.write(term + "\n")

    sampler = _make_sampler(args.vocab_size, args.zipf_s, args.seed)

    schema = pa.schema(
        [
            pa.field(args.id_column, pa.int64(), nullable=False),
            pa.field(args.text_column, pa.string(), nullable=False),
        ]
    )

    batches = _record_batches(
        rows=args.rows,
        batch_size=args.batch_size,
        vocab=vocab,
        sampler=sampler,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        seed=args.seed,
        id_column=args.id_column,
        text_column=args.text_column,
    )
    reader = pa.RecordBatchReader.from_batches(schema, batches)

    ds = lance.write_dataset(
        reader,
        args.uri,
        mode=args.mode,
        max_rows_per_file=args.max_rows_per_file,
    )

    if args.build_index:
        index_kwargs = {
            "with_position": args.with_position,
            "base_tokenizer": args.base_tokenizer,
            "language": args.language,
            "max_token_length": args.max_token_length,
            "lower_case": args.lower_case,
            "stem": args.stem,
            "remove_stop_words": args.remove_stop_words,
            "ascii_folding": args.ascii_folding,
            "prefix_only": args.prefix_only,
        }
        if args.custom_stop_words:
            index_kwargs["custom_stop_words"] = [
                w.strip() for w in args.custom_stop_words.split(",") if w.strip()
            ]
        if args.min_ngram_length is not None:
            index_kwargs["min_ngram_length"] = args.min_ngram_length
        if args.max_ngram_length is not None:
            index_kwargs["max_ngram_length"] = args.max_ngram_length

        ds.create_scalar_index(
            args.text_column,
            index_type="INVERTED",
            replace=True,
            **index_kwargs,
        )

    print("Done.")
    print(f"Dataset: {args.uri}")
    if args.emit_terms:
        print(f"Terms file: {args.emit_terms}")


if __name__ == "__main__":
    main()
