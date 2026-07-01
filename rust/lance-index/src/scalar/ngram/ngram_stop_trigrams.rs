// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stop-trigram filtering for the N-gram index.
//!
//! Ultra-high-frequency trigrams (for example `the`, `ing`, `ion`) appear in a
//! large fraction of documents. Their posting lists are large but contribute
//! little selectivity during query-time intersection. We skip indexing them and
//! treat them as unconstrained at query time.

use std::collections::HashSet;
use std::sync::LazyLock;

use super::ngram_regex::TrigramQuery;
use super::{NGRAM_N, NGRAM_TOKENIZER, ngram_to_token, tokenize_visitor};

/// Trigrams that are omitted from the index because they are too common to help
/// prune candidates. All entries must be exactly three alphanumeric characters to
/// match the fixed trigram tokenizer.
const STOP_TRIGRAMS: &[&str] = &[
    // Whole 3-letter English stop words
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one",
    "our", "out", "his", "has", "how", "its", "may", "new", "now", "old", "see", "two", "way",
    "who", "did", "get", "let", "say", "she", "too", "use",
    // Low-selectivity fragments common in English text
    "ing", "ion", "tio", "ent", "ati", "ere", "ter", "ate", "men", "est", "tha", "hat",
    "ith", // spellchecker:disable-line
    "ver", "ill", "com", "ive", "ons", "res", "ers", "nce", "lin", "ear",
    "ght", // spellchecker:disable-line
];

static STOP_TRIGRAM_TOKENS: LazyLock<HashSet<u32>> = LazyLock::new(|| {
    STOP_TRIGRAMS
        .iter()
        .map(|trigram| ngram_to_token(trigram, NGRAM_N))
        .collect()
});

/// Returns true when `token` is a stop trigram and should not be indexed or
/// required at query time.
pub fn is_stop_trigram_token(token: u32) -> bool {
    STOP_TRIGRAM_TOKENS.contains(&token)
}

/// Whether a `contains` pattern has at least one non-stop trigram the index can
/// use to prune candidates.
pub fn contains_can_use_index(substr: &str) -> bool {
    let mut has_selective_trigram = false;
    tokenize_visitor(&NGRAM_TOKENIZER, substr, |ngram| {
        let token = ngram_to_token(ngram, NGRAM_N);
        if !is_stop_trigram_token(token) {
            has_selective_trigram = true;
        }
    });
    has_selective_trigram
}

/// Remove stop-trigram requirements from a regex-derived condition.
///
/// Stop trigrams are identity for `AND` and absorbing for `OR` because we cannot
/// soundly prune on a trigram that is not stored in the index.
pub fn strip_stop_trigrams(query: TrigramQuery) -> TrigramQuery {
    match query {
        TrigramQuery::Trigram(token) if is_stop_trigram_token(token) => TrigramQuery::All,
        TrigramQuery::And(items) => {
            let stripped = items
                .into_iter()
                .map(strip_stop_trigrams)
                .filter(|item| *item != TrigramQuery::All)
                .collect();
            TrigramQuery::and(stripped)
        }
        TrigramQuery::Or(items) => {
            let stripped: Vec<_> = items.into_iter().map(strip_stop_trigrams).collect();
            if stripped.contains(&TrigramQuery::All) {
                return TrigramQuery::All;
            }
            TrigramQuery::or(stripped)
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stop_trigram_tokens_are_unique() {
        let mut seen = HashSet::new();
        for trigram in STOP_TRIGRAMS {
            assert_eq!(trigram.len(), 3, "stop trigram must be length 3: {trigram}");
            assert!(seen.insert(trigram), "duplicate stop trigram: {trigram}");
        }
    }

    #[test]
    fn test_strip_stop_trigrams_and() {
        let query = TrigramQuery::and(vec![
            TrigramQuery::Trigram(ngram_to_token("the", NGRAM_N)),
            TrigramQuery::Trigram(ngram_to_token("xyz", NGRAM_N)),
        ]);
        let stripped = strip_stop_trigrams(query);
        assert_eq!(
            stripped,
            TrigramQuery::Trigram(ngram_to_token("xyz", NGRAM_N))
        );
    }

    #[test]
    fn test_strip_stop_trigrams_or_absorbs() {
        let query = TrigramQuery::or(vec![
            TrigramQuery::Trigram(ngram_to_token("the", NGRAM_N)),
            TrigramQuery::Trigram(ngram_to_token("xyz", NGRAM_N)),
        ]);
        assert_eq!(strip_stop_trigrams(query), TrigramQuery::All);
    }

    #[test]
    fn test_contains_can_use_index_requires_selective_trigram() {
        assert!(!contains_can_use_index("the"));
        assert!(!contains_can_use_index("and"));
        assert!(contains_can_use_index("theory"));
        assert!(contains_can_use_index("uniquexyz"));
    }
}
