# Inverted Index Query: Potential Performance Optimizations

Below are candidate optimizations in the current inverted-index query path (IO, CPU, and WAND/BMW logic). I focused on the hot paths in:
- `rust/lance/src/io/exec/fts.rs`
- `rust/lance-index/src/scalar/inverted/index.rs`
- `rust/lance-index/src/scalar/inverted/wand.rs`
- `rust/lance-index/src/scalar/inverted/encoding.rs`

These are suggestions to explore; none are implemented here.

## IO / Storage

1. **Batch posting-list reads for multi‑term queries**
   - Today `PostingListReader::posting_list` reads a single token row at a time (`read_range(token_id..token_id+1)`), and `InvertedPartition::load_posting_lists` does this for each token. For fuzzy expansions or long queries this becomes many small I/Os. Consider batching contiguous token IDs (single `read_range`) or a multi‑row read API, then slicing in memory.

2. **Avoid double reads for phrase queries**
   - For phrase queries `posting_list(..., is_phrase_query=true)` first loads postings without positions, then `read_positions` does a second read. When phrase query is requested, read `POSTING_COL + POSITION_COL` in one call and cache it with a distinct key (e.g., include `with_position` in the cache key) to avoid a second I/O.

3. **Lazy / block‑level loading for very large posting lists**
   - Compressed posting lists are stored as a single row containing all blocks, so WAND still loads *all* blocks even when early termination is possible. Consider storing block offsets (or each block as a row) to enable on‑demand block reads, especially for very frequent tokens where many blocks are skipped.

4. **Partition‑level pruning before loading posting lists**
   - `InvertedIndex::bm25_search` loads posting lists for all partitions before per‑partition search. If the `RowAddrMask` (prefilter) can be mapped to partitions/fragments, you can skip partitions that cannot match any row IDs, avoiding the posting‑list reads entirely.

5. **Positions: keep compressed and decode lazily**
   - `PostingIterator::positions` fully decompresses positions into a `Vec<u32>` each time. For phrase queries with many candidates, this becomes expensive. A lazy iterator over compressed blocks (or caching per‑doc positions once) could reduce I/O and CPU for position checks.

## CPU / Scoring

1. **Precompute query weights (IDF) once per query**
   - In `InvertedIndex::bm25_search`, each candidate doc recomputes `query_weight` via `IndexBM25Scorer::num_docs_containing_token`, which scans partitions per call. For large candidate sets this is very costly. Precompute `idf` per term once (per query) and use it directly during scoring.

2. **Avoid repeated String cloning for tokens**
   - `tokens_by_position` is rebuilt per partition with cloned `String`s. `PostingIterator` also stores `String`. Consider using `Arc<str>` or storing indices into the original `Tokens` vector to reduce allocations and cache pressure (especially with fuzzy expansions).

3. **Phrase query position checks allocate each time**
   - `Wand::check_positions` allocates `Vec<PositionIterator>` and sorts it for *every* candidate doc. You can precompute the query‑term order once, reuse a small fixed buffer, and avoid resorting on every candidate.

4. **Reduce repeated decompression in tight loops**
   - `PostingIterator::doc()` is called frequently, and for compressed lists it may decompress a block repeatedly in inner loops. Consider caching the last `(block_idx, block_offset)` and avoiding redundant `doc()` calls in `next`, `check_pivot_aligned`, and `check_block_max` paths.

5. **Use token IDs for fuzzy expansions**
   - `expand_fuzzy` gets FST matches as strings, then re‑maps to token IDs. The FST already stores the token ID; use it directly to avoid string materialization and lookups for large expansions.

## WAND / BMW Algorithm

1. **Tighten block upper bounds**
   - `DocSet::calculate_block_max_scores` already multiplies by `idf * (K1+1)` when writing. In `Wand::block_max_score`, the compressed path multiplies by `(K1+1)` again. If this is redundant, bounds become too loose, reducing pruning effectiveness. Verify and remove extra scaling if safe.

2. **Avoid full sort on every `move_preceding`**
   - `Wand::move_preceding` does `postings.sort_unstable()` each time a candidate is rejected. For many terms (fuzzy OR queries) this is expensive. Maintain a heap or insertion‑sorted vector to reduce per‑candidate sort cost.

3. **Optimize pivot selection cost**
   - `find_pivot_term` recomputes a linear prefix sum of `approximate_upper_bound` for every candidate. For large term counts, keep prefix sums or update incrementally when postings move to reduce O(n) per iteration.

4. **Enable BMW on legacy / plain lists**
   - `PostingList::Plain` uses `approximate_upper_bound` for every block because there is no per‑block max. If legacy indexes are still important, consider storing block max scores there too (or compressing legacy postings on load) to enable block‑max pruning.

5. **Use global BM25 stats earlier to prune more**
   - Per‑partition WAND uses local `avgdl` and IDF. Global rescoring then discards many candidates. If you can precompute global IDF for query terms and supply it to per‑partition WAND (with correct upper bounds), you can reduce per‑partition candidate volume and CPU.

## Quick Targets (likely high ROI)

- Batch posting‑list reads for queries with many tokens (IO).
- Precompute query IDF weights once per query (CPU).
- Remove redundant scaling in `block_max_score` if verified (WAND pruning).
- Avoid per‑candidate allocation/sort in `check_positions` (phrase queries).

