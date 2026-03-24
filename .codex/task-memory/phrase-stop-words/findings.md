# Findings

- `PhraseQueryExec` tokenizes the query with the inverted index tokenizer before searching.
- The main inverted index stores per-token positions from the tokenizer stream in `rust/lance-index/src/scalar/inverted/builder.rs`.
- The in-memory WAL FTS path already assigns positions over emitted tokens only, which naturally collapses removed stop words.
- The new end-to-end Rust dataset test fails on `query=want the apple` with zero matches against docs `want the apple` / `want an apple`.
- Root cause: `collect_query_tokens` dropped tokenizer positions and `load_posting_lists` reassigned phrase query positions as dense `0..n`, so removed stop words in the query lost their gaps.
- Preserving query positions fixes the missing-match bug, but it also introduces false positives like `want green apple` because the index cannot tell whether the gap token was a stop word.
- The final fix keeps tokenizer query positions for indexed phrase matching, then performs an exact post-validation step against the original document text when the phrase query actually contains removed stop words.
