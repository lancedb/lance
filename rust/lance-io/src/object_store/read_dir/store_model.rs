// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! How an object store answers a list request, as a model the tests can vary.
//!
//! Two things a paginated listing rests on are not pinned down by any API: what a resume
//! position means, and what a page limit is spent on. Both are choices a store makes, and a
//! listing that only works for one of the choices is not a working listing. Holding them here
//! lets the fake lister in [`read_dir`](super) and the wire-level [`emulator`](super::emulator)
//! put the same store behaviours under test.

/// The largest page a store will return, whatever a request asks for. Real stores have one —
/// S3 caps `max-keys` at a thousand — and it is the reason a listing cannot page by asking for
/// ever more at once.
const DEFAULT_PAGE_BOUND: usize = 1000;

/// What a store's resume position means, which is not the same everywhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffsetMode {
    /// Keys strictly after the position, as S3's `start-after` does, and GCS's too since
    /// `object_store` lists it over the S3-compatible XML API.
    Exclusive,
    /// Keys at or after the position, as Azure's `startFrom` does.
    Inclusive,
    /// The position is dropped and the listing starts from the top, as Azurite does with
    /// `startFrom`. A listing has to stay correct and still finish, which is the point of
    /// having this here.
    Ignored,
}

/// The order a store lists its keys in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyOrder {
    /// Byte order over whole keys, which is what a flat namespace gives: a key is a name and
    /// names sort against each other whole.
    ByKey,
    /// Byte order with the delimiter below every other character. A store that holds a real
    /// directory tree orders each level by child name, so a directory comes before a sibling
    /// whose name extends it — `foo/` before `foo-bar/`, where byte order has them the other
    /// way round. Azure documents this for accounts with a hierarchical namespace.
    DelimiterLowest,
    /// No order a caller could predict, which is what S3 Express gives. Spelled as byte order
    /// backwards so that a test is repeatable, but nothing may rely on the shape of it.
    Reversed,
}

impl KeyOrder {
    fn cmp(&self, left: &str, right: &str) -> std::cmp::Ordering {
        match self {
            Self::ByKey => left.cmp(right),
            Self::DelimiterLowest => left
                .bytes()
                .map(Self::rank)
                .cmp(right.bytes().map(Self::rank)),
            Self::Reversed => right.cmp(left),
        }
    }

    /// The delimiter moved below every byte that sorts under it, and nothing else disturbed.
    fn rank(byte: u8) -> u8 {
        match byte {
            b'/' => 0,
            byte if byte < b'/' => byte + 1,
            byte => byte,
        }
    }
}

/// What a page limit is spent on, which the S3 API does not pin down.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetMode {
    /// The limit counts the entries in the page, so a collapsed prefix costs one.
    PerEntry,
    /// The limit counts the keys the store scans, so a collapsed prefix costs every key it
    /// collapsed. A page can then come back holding one entry and still be truncated.
    PerScannedKey,
}

/// The keys a store holds, and how it behaves when listing them.
#[derive(Debug, Clone)]
pub struct StoreModel {
    /// Every key in the store, in the order the store would return them.
    keys: Vec<String>,
    offset: OffsetMode,
    budget: BudgetMode,
    order: KeyOrder,
    page_bound: usize,
}

impl StoreModel {
    pub fn new<K: Into<String>>(keys: impl IntoIterator<Item = K>) -> Self {
        let keys: Vec<String> = keys.into_iter().map(Into::into).collect();
        Self {
            keys,
            offset: OffsetMode::Exclusive,
            budget: BudgetMode::PerEntry,
            order: KeyOrder::ByKey,
            page_bound: DEFAULT_PAGE_BOUND,
        }
        .sorted()
    }

    fn sorted(mut self) -> Self {
        let order = self.order;
        self.keys.sort_by(|left, right| order.cmp(left, right));
        self
    }

    pub fn with_offset(mut self, offset: OffsetMode) -> Self {
        self.offset = offset;
        self
    }

    pub fn with_order(mut self, order: KeyOrder) -> Self {
        self.order = order;
        self.sorted()
    }

    pub fn with_budget(mut self, budget: BudgetMode) -> Self {
        self.budget = budget;
        self
    }

    /// Cap pages at `page_bound` entries, so that a directory holding more than that cannot be
    /// listed in one request however large a page the caller asks for.
    pub fn with_page_bound(mut self, page_bound: usize) -> Self {
        self.page_bound = page_bound;
        self
    }

    /// One page of the immediate children of `prefix`.
    ///
    /// `resume` is a position within the listing: a continuation token if the request
    /// carried one, which is exact and always exclusive, otherwise the caller's offset,
    /// which [`OffsetMode`] interprets.
    ///
    /// The delimiter is always applied. Every request this crate makes sets one, and a
    /// recursive listing would not exercise anything a paginated directory listing does.
    pub fn list_level(
        &self,
        prefix: &str,
        resume: Resume<'_>,
        max_keys: Option<usize>,
    ) -> LevelPage {
        let mut page = LevelPage::default();
        let mut budget = max_keys.unwrap_or(self.page_bound).min(self.page_bound);
        let mut idx = 0;

        while idx < self.keys.len() {
            let key = &self.keys[idx];
            let Some(rest) = key.strip_prefix(prefix) else {
                idx += 1;
                continue;
            };
            if resume.consumes(key, self.offset, self.order) {
                idx += 1;
                continue;
            }
            if budget == 0 {
                page.truncated = true;
                // A real token is opaque and resumes exactly where the page stopped. The
                // last key this page consumed says the same thing.
                page.next_token = Some(self.keys[idx - 1].clone());
                break;
            }
            match rest.find('/') {
                Some(end) => {
                    let child = format!("{prefix}{}/", &rest[..end]);
                    page.prefixes.push(child.clone());
                    while idx < self.keys.len() && self.keys[idx].starts_with(&child) {
                        idx += 1;
                        if self.budget == BudgetMode::PerScannedKey {
                            budget = budget.saturating_sub(1);
                        }
                    }
                    if self.budget == BudgetMode::PerEntry {
                        budget -= 1;
                    }
                }
                None => {
                    page.objects.push(key.clone());
                    idx += 1;
                    budget -= 1;
                }
            }
        }

        page
    }
}

/// Where a list request asks the store to pick up from.
#[derive(Debug, Clone, Copy)]
pub enum Resume<'a> {
    Start,
    /// The caller's offset, whose meaning is the store's to decide.
    Offset(&'a str),
    /// The store's own continuation token, which it always honours exactly.
    Token(&'a str),
}

impl Resume<'_> {
    fn consumes(&self, key: &str, offset: OffsetMode, order: KeyOrder) -> bool {
        use std::cmp::Ordering::{Equal, Less};
        match self {
            Self::Start => false,
            Self::Token(token) => order.cmp(key, token) != std::cmp::Ordering::Greater,
            Self::Offset(offset_key) => match offset {
                OffsetMode::Exclusive => matches!(order.cmp(key, offset_key), Less | Equal),
                OffsetMode::Inclusive => order.cmp(key, offset_key) == Less,
                OffsetMode::Ignored => false,
            },
        }
    }
}

/// One page of a directory level, as the store would report it.
#[derive(Debug, Default)]
pub struct LevelPage {
    /// Child directories, as keys ending in the delimiter.
    pub prefixes: Vec<String>,
    /// Child objects, as whole keys.
    pub objects: Vec<String>,
    pub truncated: bool,
    pub next_token: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    /// Two tables and a loose file, so a page can hold a collapsed prefix and an object.
    const KEYS: &[&str] = &[
        "db/a.lance/data/1.lance",
        "db/a.lance/data/2.lance",
        "db/b.lance/data/1.lance",
        "db/loose.txt",
    ];

    fn model(offset: OffsetMode, budget: BudgetMode) -> StoreModel {
        StoreModel::new(KEYS.to_vec())
            .with_offset(offset)
            .with_budget(budget)
    }

    /// What an offset excludes is the store's choice, and all three choices are real: S3 and
    /// GCS exclude the position itself, Azure includes it, and Azurite ignores it. A file key
    /// shows the difference; a directory's prefix cannot, since the keys inside it sort after
    /// it and come back under every mode.
    #[rstest]
    #[case::exclusive(OffsetMode::Exclusive, vec![], vec![])]
    #[case::inclusive(OffsetMode::Inclusive, vec![], vec!["db/loose.txt"])]
    #[case::ignored(
        OffsetMode::Ignored,
        vec!["db/a.lance/", "db/b.lance/"],
        vec!["db/loose.txt"]
    )]
    fn test_offset_modes_differ_at_the_position_itself(
        #[case] offset: OffsetMode,
        #[case] prefixes: Vec<&str>,
        #[case] objects: Vec<&str>,
    ) {
        let page = model(offset, BudgetMode::PerEntry).list_level(
            "db/",
            Resume::Offset("db/loose.txt"),
            Some(4),
        );
        assert_eq!(page.prefixes, prefixes);
        assert_eq!(page.objects, objects);
    }

    /// A collapsed prefix costs one entry on a store that counts entries and everything it
    /// collapsed on a store that counts scanned keys. The second kind returns a page that is
    /// short and truncated at once, which is why a short page cannot end a listing.
    #[rstest]
    #[case::per_entry(BudgetMode::PerEntry, vec!["db/a.lance/", "db/b.lance/"])]
    #[case::per_scanned_key(BudgetMode::PerScannedKey, vec!["db/a.lance/"])]
    fn test_budget_modes_differ_over_a_collapsed_prefix(
        #[case] budget: BudgetMode,
        #[case] expected: Vec<&str>,
    ) {
        let page = model(OffsetMode::Exclusive, budget).list_level("db/", Resume::Start, Some(2));
        assert_eq!(page.prefixes, expected);
        assert!(page.truncated, "loose.txt is still to come");
    }

    /// A continuation token is the store's own position and is exact, so it resumes the same
    /// way whatever the store does with a caller-supplied offset.
    #[rstest]
    fn test_a_token_resumes_exactly(
        #[values(OffsetMode::Exclusive, OffsetMode::Inclusive, OffsetMode::Ignored)]
        offset: OffsetMode,
    ) {
        let model = model(offset, BudgetMode::PerScannedKey);
        let first = model.list_level("db/", Resume::Start, Some(1));
        assert_eq!(first.prefixes, vec!["db/a.lance/"]);
        assert!(first.truncated);

        let token = first
            .next_token
            .expect("a truncated page hands back a token");
        let second = model.list_level("db/", Resume::Token(&token), Some(2));
        assert_eq!(second.prefixes, vec!["db/b.lance/"]);
        assert_eq!(second.objects, vec!["db/loose.txt"]);
        assert!(!second.truncated);
    }
}
