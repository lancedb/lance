// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Idempotency keys: duplicate detection within a commit's read-version window.
//!
//! A writer attaches work ids to detect concurrent duplicate writes and
//! retries that preserve the original read version, through
//! [`CommitBuilder::with_idempotency_keys`] or
//! [`WriteParams::with_idempotency_keys`]. They are stored in the reserved
//! transaction property [`IDEMPOTENCY_KEYS_PROPERTY`] as a JSON array of
//! strings. The conflict resolver treats two transactions whose keys
//! intersect as incompatible: the later commit fails with
//! [`Error::DuplicateTransaction`] instead of rebasing over the transaction
//! that already applied the work.
//!
//! # Window
//!
//! A commit checks every transaction committed after its read version, and
//! the manifest put is create-if-absent, so a successful commit has checked
//! every version after its read version up to the version it created.
//! Versions at or before the read version are not checked. A fresh read
//! after an earlier commit can therefore let the same work commit again.
//! The caller must detect work in those earlier versions separately, or
//! preserve a read version from before the first commit and retain the
//! required history. These keys alone do not provide table-wide at-most-once
//! execution.
//!
//! # Experimental
//!
//! Idempotency keys are an experimental format feature. Release builds
//! reject keys unless [`ENABLE_UNSTABLE_IDEMPOTENCY_KEYS_ENV`] is set; debug
//! builds always accept them so tests exercise the path. A build that rejects
//! keys commits exactly as it did before keys existed.
//!
//! # Caller-named property
//!
//! [`CommitBuilder::with_idempotency_property`] names a transaction property
//! the caller already writes its keys to (a JSON array of strings). The keys
//! are copied into [`IDEMPOTENCY_KEYS_PROPERTY`] as well, and other
//! transactions' keys are read from both properties, so writers using either
//! API on one table see each other's keys.
//!
//! [`CommitBuilder::with_idempotency_keys`]: crate::dataset::CommitBuilder::with_idempotency_keys
//! [`CommitBuilder::with_idempotency_property`]: crate::dataset::CommitBuilder::with_idempotency_property
//! [`WriteParams::with_idempotency_keys`]: crate::dataset::WriteParams::with_idempotency_keys

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, OnceLock};

use lance_core::{Error, Result};

use crate::dataset::transaction::Transaction;

/// Reserved transaction property holding a transaction's idempotency keys as
/// a JSON array of strings.
pub const IDEMPOTENCY_KEYS_PROPERTY: &str = "__lance_idempotency_keys";

/// Default for the most distinct idempotency keys one transaction may carry;
/// override with [`MAX_IDEMPOTENCY_KEYS_ENV`].
///
/// A keyed commit compares its keys with those of every transaction committed
/// since its read version, and the keys are persisted with the transaction,
/// so every later conflict check in the window reads them again. The limits
/// keep that cost bounded while leaving room for a batch of a few thousand
/// work items (e.g. queue messages) per commit.
pub const DEFAULT_MAX_IDEMPOTENCY_KEYS: usize = 4096;

/// Default for the most bytes of idempotency keys (UTF-8, summed over the
/// distinct keys) one transaction may carry; override with
/// [`MAX_IDEMPOTENCY_KEY_BYTES_ENV`]. [`DEFAULT_MAX_IDEMPOTENCY_KEYS`] keys of
/// 64 bytes, enough for UUIDs or queue message ids with a prefix.
pub const DEFAULT_MAX_IDEMPOTENCY_KEY_BYTES: usize = 256 * 1024;

/// Env var overriding [`DEFAULT_MAX_IDEMPOTENCY_KEYS`] process-wide, as a
/// positive integer.
pub const MAX_IDEMPOTENCY_KEYS_ENV: &str = "LANCE_MAX_IDEMPOTENCY_KEYS";

/// Env var overriding [`DEFAULT_MAX_IDEMPOTENCY_KEY_BYTES`] process-wide, as
/// a positive integer.
pub const MAX_IDEMPOTENCY_KEY_BYTES_ENV: &str = "LANCE_MAX_IDEMPOTENCY_KEY_BYTES";

/// Env var that opts a release build into idempotency keys before the format
/// feature is stabilized.
pub const ENABLE_UNSTABLE_IDEMPOTENCY_KEYS_ENV: &str = "LANCE_ENABLE_UNSTABLE_IDEMPOTENCY_KEYS";

/// Whether this build accepts idempotency keys: always in debug builds, and in
/// release builds only when [`ENABLE_UNSTABLE_IDEMPOTENCY_KEYS_ENV`] is set,
/// like the unstable features in `lance_table::feature_flags`.
fn idempotency_keys_enabled() -> bool {
    cfg!(debug_assertions) || std::env::var_os(ENABLE_UNSTABLE_IDEMPOTENCY_KEYS_ENV).is_some()
}

/// Split out from [`idempotency_keys_enabled`] so the rejection is testable
/// without toggling the build profile or environment.
fn require_enabled(enabled: bool) -> Result<()> {
    if enabled {
        Ok(())
    } else {
        Err(Error::not_supported(format!(
            "idempotency keys are an experimental feature; set \
             {ENABLE_UNSTABLE_IDEMPOTENCY_KEYS_ENV} to use them"
        )))
    }
}

/// The most distinct idempotency keys one transaction may carry:
/// [`MAX_IDEMPOTENCY_KEYS_ENV`] if valid, otherwise
/// [`DEFAULT_MAX_IDEMPOTENCY_KEYS`]. Read once per process.
pub(crate) fn max_idempotency_keys() -> usize {
    static LIMIT: OnceLock<usize> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        parse_limit(
            MAX_IDEMPOTENCY_KEYS_ENV,
            std::env::var(MAX_IDEMPOTENCY_KEYS_ENV).ok().as_deref(),
            DEFAULT_MAX_IDEMPOTENCY_KEYS,
        )
    })
}

/// The most bytes of idempotency keys one transaction may carry:
/// [`MAX_IDEMPOTENCY_KEY_BYTES_ENV`] if valid, otherwise
/// [`DEFAULT_MAX_IDEMPOTENCY_KEY_BYTES`]. Read once per process.
pub(crate) fn max_idempotency_key_bytes() -> usize {
    static LIMIT: OnceLock<usize> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        parse_limit(
            MAX_IDEMPOTENCY_KEY_BYTES_ENV,
            std::env::var(MAX_IDEMPOTENCY_KEY_BYTES_ENV).ok().as_deref(),
            DEFAULT_MAX_IDEMPOTENCY_KEY_BYTES,
        )
    })
}

/// Parse the value of the limit env var `name`; `None` means unset. Anything
/// that is not a positive integer warns and falls back to `default`.
fn parse_limit(name: &str, raw: Option<&str>, default: usize) -> usize {
    let Some(raw) = raw else {
        return default;
    };
    match raw.trim().parse::<usize>() {
        Ok(limit) if limit > 0 => limit,
        _ => {
            log::warn!("ignoring invalid {name}={raw:?}; using the default of {default}");
            default
        }
    }
}

/// The idempotency keys of the transaction being committed, and where to look
/// for the keys of the transactions it is checked against.
#[derive(Debug, Clone)]
pub(crate) struct IdempotencyKeys {
    keys: BTreeSet<String>,
    /// A property set by [`CommitBuilder::with_idempotency_property`]: other
    /// transactions may carry keys there, written by writers that only know
    /// that property.
    ///
    /// [`CommitBuilder::with_idempotency_property`]: crate::dataset::CommitBuilder::with_idempotency_property
    caller_property: Option<String>,
}

impl IdempotencyKeys {
    /// The keys of `transaction`, or `None` when it carries none.
    ///
    /// Fails when the keys are malformed or exceed the limits, or when this
    /// build does not enable the feature, so a commit whose keys cannot be
    /// checked is rejected before it does any IO. Every commit path calls
    /// this, so it is also where the experimental gate is enforced.
    pub(crate) fn from_transaction(transaction: &Transaction) -> Result<Option<Self>> {
        let keys = transaction_keys(transaction)?;
        if keys.is_empty() {
            return Ok(None);
        }
        validate(&keys)?;
        Ok(Some(Self {
            keys,
            caller_property: None,
        }))
    }

    /// Also read other transactions' keys from `property`.
    pub(crate) fn with_caller_property(mut self, property: Option<&str>) -> Self {
        self.caller_property = property
            .filter(|property| *property != IDEMPOTENCY_KEYS_PROPERTY)
            .map(str::to_string);
        self
    }

    /// Fail when `other`, committed at `other_version`, carries any of our keys.
    ///
    /// A transaction without keys (compaction, index builds, writers that do
    /// not use idempotency keys) never conflicts. A malformed value fails the
    /// commit, since it cannot be checked.
    pub(crate) fn check(&self, other: &Transaction, other_version: u64) -> Result<()> {
        let mut overlap = BTreeSet::new();
        for name in
            std::iter::once(IDEMPOTENCY_KEYS_PROPERTY).chain(self.caller_property.as_deref())
        {
            let Some(raw) = property(other, name) else {
                continue;
            };
            let theirs = parse_keys(raw).map_err(|err| {
                Error::invalid_input(format!(
                    "cannot check idempotency keys against the transaction committed at \
                     version {other_version}: its property '{name}' is not a JSON array of \
                     strings: {err}"
                ))
            })?;
            overlap.extend(self.keys.intersection(&theirs).cloned());
        }
        if overlap.is_empty() {
            Ok(())
        } else {
            Err(Error::duplicate_transaction(
                other_version,
                overlap.into_iter().collect(),
            ))
        }
    }
}

/// The idempotency keys `transaction` carries, empty when it has none.
pub(crate) fn transaction_keys(transaction: &Transaction) -> Result<BTreeSet<String>> {
    match property(transaction, IDEMPOTENCY_KEYS_PROPERTY) {
        Some(raw) => parse_keys(raw).map_err(|err| malformed(IDEMPOTENCY_KEYS_PROPERTY, err)),
        None => Ok(BTreeSet::new()),
    }
}

/// Add `keys` to the idempotency keys `transaction` already carries.
///
/// Leaves the transaction untouched when there are no keys at all, so a
/// commit without keys persists exactly what it did before keys existed.
pub(crate) fn add_keys(
    transaction: &mut Transaction,
    keys: impl IntoIterator<Item = String>,
) -> Result<()> {
    let mut all = transaction_keys(transaction)?;
    all.extend(keys);
    if all.is_empty() {
        return Ok(());
    }
    let mut properties = transaction
        .transaction_properties
        .as_deref()
        .cloned()
        .unwrap_or_default();
    properties.insert(IDEMPOTENCY_KEYS_PROPERTY.to_string(), encode_keys(all));
    transaction.transaction_properties = Some(Arc::new(properties));
    Ok(())
}

/// The keys a caller wrote to `property` of the transaction being committed.
///
/// Fails when the property is missing: the caller asked for the check, and a
/// transaction without keys would silently commit unchecked.
pub(crate) fn caller_property_keys(
    transaction: &Transaction,
    property_name: &str,
) -> Result<BTreeSet<String>> {
    let raw = property(transaction, property_name).ok_or_else(|| {
        Error::invalid_input(format!(
            "idempotency property '{property_name}' is not set on the transaction being committed"
        ))
    })?;
    parse_keys(raw).map_err(|err| malformed(property_name, err))
}

/// Encode keys as the JSON array stored in [`IDEMPOTENCY_KEYS_PROPERTY`].
pub(crate) fn encode_keys(keys: impl IntoIterator<Item = String>) -> String {
    serde_json::Value::from(keys.into_iter().collect::<Vec<_>>()).to_string()
}

/// Insert `keys` as the idempotency keys in `properties`, replacing any there.
pub(crate) fn set_keys(properties: &mut HashMap<String, String>, keys: Vec<String>) {
    if keys.is_empty() {
        properties.remove(IDEMPOTENCY_KEYS_PROPERTY);
    } else {
        properties.insert(IDEMPOTENCY_KEYS_PROPERTY.to_string(), encode_keys(keys));
    }
}

fn property<'a>(transaction: &'a Transaction, name: &str) -> Option<&'a str> {
    transaction
        .transaction_properties
        .as_ref()
        .and_then(|properties| properties.get(name))
        .map(String::as_str)
}

fn parse_keys(raw: &str) -> std::result::Result<BTreeSet<String>, serde_json::Error> {
    serde_json::from_str::<Vec<String>>(raw).map(|keys| keys.into_iter().collect())
}

fn malformed(property: &str, err: serde_json::Error) -> Error {
    Error::invalid_input(format!(
        "idempotency property '{property}' is not a JSON array of strings: {err}"
    ))
}

/// Check the idempotency keys in a write's transaction `properties` before
/// any data is written, so a write that cannot commit writes no data files.
pub(crate) fn validate_properties(properties: &HashMap<String, String>) -> Result<()> {
    let Some(raw) = properties.get(IDEMPOTENCY_KEYS_PROPERTY) else {
        return Ok(());
    };
    let keys = parse_keys(raw).map_err(|err| malformed(IDEMPOTENCY_KEYS_PROPERTY, err))?;
    if keys.is_empty() {
        return Ok(());
    }
    validate(&keys)
}

fn validate(keys: &BTreeSet<String>) -> Result<()> {
    require_enabled(idempotency_keys_enabled())?;
    validate_with_limits(keys, max_idempotency_keys(), max_idempotency_key_bytes())
}

fn validate_with_limits(
    keys: &BTreeSet<String>,
    max_keys: usize,
    max_key_bytes: usize,
) -> Result<()> {
    if keys.contains("") {
        return Err(Error::invalid_input(
            "idempotency keys must be non-empty strings, got an empty key",
        ));
    }
    if keys.len() > max_keys {
        return Err(Error::invalid_input(format!(
            "a transaction may carry at most {max_keys} idempotency keys, got {}; set \
             {MAX_IDEMPOTENCY_KEYS_ENV} to change the limit",
            keys.len()
        )));
    }
    let total_bytes: usize = keys.iter().map(String::len).sum();
    if total_bytes > max_key_bytes {
        return Err(Error::invalid_input(format!(
            "a transaction's idempotency keys may total at most {max_key_bytes} bytes, got \
             {total_bytes} bytes in {} keys; set {MAX_IDEMPOTENCY_KEY_BYTES_ENV} to change the \
             limit",
            keys.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use arrow_array::types::Int32Type;
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen::{array, gen_batch};
    use lance_table::transaction::resolve_arrow_field_ids;
    use object_store::ObjectStoreExt;
    use rstest::rstest;

    use super::*;
    use crate::Dataset;
    use crate::dataset::transaction::Operation;
    use crate::dataset::{CommitBuilder, WriteMode, WriteParams};
    use crate::utils::test::{
        AmbiguousCommitHandler, AmbiguousFailure, DatagenExt, FragmentCount, FragmentRowCount,
    };

    const CALLER_PROPERTY: &str = "app_keys";

    fn transaction(
        read_version: u64,
        operation: Operation,
        properties: &[(&str, &str)],
    ) -> Transaction {
        let mut transaction = Transaction::new(read_version, operation, None);
        if !properties.is_empty() {
            transaction.transaction_properties = Some(Arc::new(
                properties
                    .iter()
                    .map(|(name, value)| (name.to_string(), value.to_string()))
                    .collect(),
            ));
        }
        transaction
    }

    fn append(read_version: u64, properties: &[(&str, &str)]) -> Transaction {
        transaction(
            read_version,
            Operation::Append { fragments: vec![] },
            properties,
        )
    }

    fn overwrite(dataset: &Dataset, read_version: u64) -> Transaction {
        transaction(
            read_version,
            Operation::Overwrite {
                schema: dataset.schema().clone(),
                fragments: vec![],
                config_upsert_values: None,
                initial_bases: None,
            },
            &[],
        )
    }

    async fn dataset() -> Arc<Dataset> {
        Arc::new(
            gen_batch()
                .col("i", array::step::<Int32Type>())
                .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(10))
                .await
                .unwrap(),
        )
    }

    /// Commit `transaction` against `dataset` with `keys`.
    async fn commit_keyed(
        dataset: &Arc<Dataset>,
        transaction: Transaction,
        keys: &[&str],
    ) -> Result<Dataset> {
        CommitBuilder::new(dataset.clone())
            .with_idempotency_keys(keys.iter().copied())
            .execute(transaction)
            .await
    }

    fn assert_duplicate(result: Result<Dataset>, expected_version: u64, expected_keys: &[&str]) {
        match result {
            Err(Error::DuplicateTransaction { version, keys, .. }) => {
                assert_eq!(version, expected_version);
                assert_eq!(keys, expected_keys);
            }
            other => panic!("expected DuplicateTransaction, got {other:?}"),
        }
    }

    fn assert_invalid_input(result: Result<impl std::fmt::Debug>, message: &str) {
        match result {
            Err(err @ Error::InvalidInput { .. }) => assert!(
                err.to_string().contains(message),
                "expected '{message}' in: {err}"
            ),
            other => panic!("expected InvalidInput containing '{message}', got {other:?}"),
        }
    }

    fn stored_keys(transaction: &Transaction, property_name: &str) -> Option<String> {
        property(transaction, property_name).map(str::to_string)
    }

    #[rstest]
    #[case::disjoint(&[(IDEMPOTENCY_KEYS_PROPERTY, r#"["b"]"#)], None, Ok(vec![]))]
    #[case::overlap(&[(IDEMPOTENCY_KEYS_PROPERTY, r#"["a","b"]"#)], None, Ok(vec!["a"]))]
    #[case::no_properties(&[], None, Ok(vec![]))]
    #[case::other_property(&[("unrelated", r#"["a"]"#)], None, Ok(vec![]))]
    #[case::caller_property_not_configured(&[(CALLER_PROPERTY, r#"["a"]"#)], None, Ok(vec![]))]
    #[case::caller_property(&[(CALLER_PROPERTY, r#"["a"]"#)], Some(CALLER_PROPERTY), Ok(vec!["a"]))]
    #[case::both_properties(
        &[(IDEMPOTENCY_KEYS_PROPERTY, r#"["a"]"#), (CALLER_PROPERTY, r#"["c","d"]"#)],
        Some(CALLER_PROPERTY),
        Ok(vec!["a", "c"])
    )]
    #[case::malformed(&[(IDEMPOTENCY_KEYS_PROPERTY, "not json")], None, Err("version 7"))]
    #[case::not_strings(&[(IDEMPOTENCY_KEYS_PROPERTY, "[1]")], None, Err("version 7"))]
    #[case::malformed_caller_property(
        &[(CALLER_PROPERTY, r#"{"a":1}"#)],
        Some(CALLER_PROPERTY),
        Err(CALLER_PROPERTY)
    )]
    fn test_check(
        #[case] other_properties: &[(&str, &str)],
        #[case] caller_property: Option<&str>,
        #[case] expected: std::result::Result<Vec<&str>, &str>,
    ) {
        let ours = append(1, &[(IDEMPOTENCY_KEYS_PROPERTY, r#"["a","c"]"#)]);
        let keys = IdempotencyKeys::from_transaction(&ours)
            .unwrap()
            .unwrap()
            .with_caller_property(caller_property);
        let result = keys.check(&append(1, other_properties), 7);
        match expected {
            Ok(overlap) if overlap.is_empty() => result.unwrap(),
            Ok(overlap) => match result {
                Err(Error::DuplicateTransaction { version, keys, .. }) => {
                    assert_eq!(version, 7);
                    assert_eq!(keys, overlap);
                }
                other => panic!("expected DuplicateTransaction, got {other:?}"),
            },
            Err(message) => assert_invalid_input(result, message),
        }
    }

    #[rstest]
    #[case::missing(&[], Ok(None))]
    #[case::empty_array(&[(IDEMPOTENCY_KEYS_PROPERTY, "[]")], Ok(None))]
    #[case::keys(&[(IDEMPOTENCY_KEYS_PROPERTY, r#"["b","a","b"]"#)], Ok(Some(vec!["a", "b"])))]
    #[case::malformed(&[(IDEMPOTENCY_KEYS_PROPERTY, "[\"a\"")], Err("not a JSON array"))]
    #[case::empty_key(&[(IDEMPOTENCY_KEYS_PROPERTY, r#"["a",""]"#)], Err("non-empty"))]
    fn test_from_transaction(
        #[case] properties: &[(&str, &str)],
        #[case] expected: std::result::Result<Option<Vec<&str>>, &str>,
    ) {
        let result = IdempotencyKeys::from_transaction(&append(1, properties));
        match expected {
            Ok(expected) => assert_eq!(
                result
                    .unwrap()
                    .map(|keys| keys.keys.into_iter().collect::<Vec<_>>()),
                expected.map(|keys| keys.into_iter().map(str::to_string).collect())
            ),
            Err(message) => assert_invalid_input(result, message),
        }
    }

    /// The default limits apply when the env vars are unset (as in tests).
    #[rstest]
    #[case::too_many_keys(
        (0..=DEFAULT_MAX_IDEMPOTENCY_KEYS).map(|i| i.to_string()).collect(),
        "at most 4096 idempotency keys, got 4097; set LANCE_MAX_IDEMPOTENCY_KEYS"
    )]
    #[case::too_many_bytes(
        (0..3)
            .map(|i| format!("{i}{}", "x".repeat(DEFAULT_MAX_IDEMPOTENCY_KEY_BYTES / 3)))
            .collect(),
        "at most 262144 bytes, got 262146 bytes in 3 keys; set LANCE_MAX_IDEMPOTENCY_KEY_BYTES"
    )]
    fn test_limits(#[case] keys: Vec<String>, #[case] message: &str) {
        let encoded = encode_keys(keys);
        let transaction = append(1, &[(IDEMPOTENCY_KEYS_PROPERTY, &encoded)]);
        assert_invalid_input(IdempotencyKeys::from_transaction(&transaction), message);
    }

    #[test]
    fn test_require_enabled() {
        require_enabled(true).unwrap();
        let err = require_enabled(false).unwrap_err();
        assert!(matches!(err, Error::NotSupported { .. }), "{err:?}");
        assert!(
            err.to_string()
                .contains(ENABLE_UNSTABLE_IDEMPOTENCY_KEYS_ENV),
            "{err}"
        );
    }

    #[test]
    fn test_limits_are_inclusive() {
        let keys = (0..DEFAULT_MAX_IDEMPOTENCY_KEYS)
            .map(|i| format!("{i:064}"))
            .collect::<Vec<_>>();
        assert_eq!(
            keys.iter().map(String::len).sum::<usize>(),
            DEFAULT_MAX_IDEMPOTENCY_KEY_BYTES
        );
        let encoded = encode_keys(keys);
        let transaction = append(1, &[(IDEMPOTENCY_KEYS_PROPERTY, &encoded)]);
        assert!(
            IdempotencyKeys::from_transaction(&transaction)
                .unwrap()
                .is_some()
        );
    }

    #[rstest]
    #[case::raised(3, 100, Ok(()))]
    #[case::fewer_keys(2, 100, Err("at most 2 idempotency keys, got 3"))]
    #[case::fewer_bytes(3, 2, Err("at most 2 bytes, got 3 bytes in 3 keys"))]
    fn test_configured_limits(
        #[case] max_keys: usize,
        #[case] max_key_bytes: usize,
        #[case] expected: std::result::Result<(), &str>,
    ) {
        let keys = ["a", "b", "c"].map(str::to_string).into_iter().collect();
        let result = validate_with_limits(&keys, max_keys, max_key_bytes);
        match expected {
            Ok(()) => result.unwrap(),
            Err(message) => assert_invalid_input(result, message),
        }
    }

    #[rstest]
    #[case::unset(None, 7)]
    #[case::value(Some("100000"), 100000)]
    #[case::padded(Some(" 12 "), 12)]
    #[case::zero(Some("0"), 7)]
    #[case::negative(Some("-1"), 7)]
    #[case::not_a_number(Some("many"), 7)]
    #[case::empty(Some(""), 7)]
    fn test_parse_limit(#[case] raw: Option<&str>, #[case] expected: usize) {
        assert_eq!(parse_limit("LANCE_TEST_LIMIT", raw, 7), expected);
    }

    /// A transaction committed after the read version that shares a key
    /// fails the commit whatever the operation, including an overwrite that
    /// otherwise rebases over everything.
    #[rstest]
    #[case::append_overlap(false, &["b", "c"], Some(vec!["b"]))]
    #[case::overwrite_overlap(true, &["c", "a"], Some(vec!["a"]))]
    #[case::append_disjoint(false, &["c"], None)]
    #[case::overwrite_disjoint(true, &["c"], None)]
    #[tokio::test]
    async fn test_shared_key_fails_commit(
        #[case] is_overwrite: bool,
        #[case] keys: &[&str],
        #[case] expected_overlap: Option<Vec<&str>>,
    ) {
        let original = dataset().await;
        let committed = commit_keyed(&original, append(1, &[]), &["a", "b"])
            .await
            .unwrap();
        assert_eq!(committed.manifest.version, 2);

        let ours = if is_overwrite {
            overwrite(&original, 1)
        } else {
            append(1, &[])
        };
        let result = commit_keyed(&original, ours, keys).await;
        match expected_overlap {
            Some(overlap) => assert_duplicate(result, 2, &overlap),
            None => assert_eq!(result.unwrap().manifest.version, 3),
        }
    }

    #[tokio::test]
    async fn test_keys_are_persisted() {
        let original = dataset().await;
        let keyed = commit_keyed(&original, append(1, &[("note", "kept")]), &["y", "x", "y"])
            .await
            .unwrap();
        let persisted = keyed.read_transaction().await.unwrap().unwrap();
        assert_eq!(
            stored_keys(&persisted, IDEMPOTENCY_KEYS_PROPERTY).as_deref(),
            Some(r#"["x","y"]"#)
        );
        assert_eq!(stored_keys(&persisted, "note").as_deref(), Some("kept"));

        // A commit without keys persists exactly what it was given.
        let unkeyed = CommitBuilder::new(Arc::new(keyed))
            .execute(append(2, &[]))
            .await
            .unwrap();
        let persisted = unkeyed.read_transaction().await.unwrap().unwrap();
        assert_eq!(persisted.transaction_properties, None);
    }

    /// Versions at or before the read version are not checked; a blind
    /// overwrite (read version 0) is checked from the version it loaded.
    #[rstest]
    #[case::read_version_is_keyed_version(false, false, None)]
    #[case::blind_overwrite_from_stale_handle(true, true, Some(2))]
    #[case::blind_overwrite_from_current_handle(true, false, None)]
    #[tokio::test]
    async fn test_window_starts_after_read_version(
        #[case] is_blind_overwrite: bool,
        #[case] is_stale_handle: bool,
        #[case] expected_duplicate: Option<u64>,
    ) {
        let original = dataset().await;
        let current = Arc::new(
            commit_keyed(&original, append(1, &[]), &["a"])
                .await
                .unwrap(),
        );
        let handle = if is_stale_handle { &original } else { &current };
        let ours = if is_blind_overwrite {
            overwrite(&original, 0)
        } else {
            append(2, &[])
        };
        let result = commit_keyed(handle, ours, &["a"]).await;
        match expected_duplicate {
            Some(version) => assert_duplicate(result, version, &["a"]),
            None => assert_eq!(result.unwrap().manifest.version, 3),
        }
    }

    /// A retryable conflict with an earlier version of the window must not
    /// mask a duplicate in a later one: a caller retrying the conflict would
    /// re-read past the duplicate.
    #[rstest]
    #[case::keyed(true)]
    #[case::unkeyed(false)]
    #[tokio::test]
    async fn test_duplicate_takes_precedence_over_other_conflicts(#[case] is_keyed: bool) {
        let original = dataset().await;
        let mut updated = original.as_ref().clone();
        updated.update_schema_metadata([("k", "v")]).await.unwrap();
        commit_keyed(&Arc::new(updated), append(2, &[]), &["a"])
            .await
            .unwrap();

        let mut operation = Operation::Merge {
            schema: original.schema().clone(),
            fragments: original.manifest.fragments.as_ref().clone(),
            preserves_nullability: true,
        };
        resolve_arrow_field_ids(Some(&original.manifest), &mut operation).unwrap();
        let ours = transaction(1, operation, &[]);
        let keys: &[&str] = if is_keyed { &["a"] } else { &[] };
        let result = commit_keyed(&original, ours, keys).await;
        if is_keyed {
            assert_duplicate(result, 3, &["a"]);
        } else {
            assert!(
                matches!(
                    result,
                    Err(Error::RetryableCommitConflict { version: 2, .. })
                ),
                "{result:?}"
            );
        }
    }

    /// Versions of the window that were cleaned up cannot be checked, so a
    /// keyed commit fails closed rather than skipping them.
    #[tokio::test]
    async fn test_cleaned_up_window_fails_closed() {
        let original = dataset().await;
        let mut latest = Arc::new(original.as_ref().clone());
        for read_version in 1..=2 {
            latest = Arc::new(
                CommitBuilder::new(latest)
                    .execute(append(read_version, &[]))
                    .await
                    .unwrap(),
            );
        }
        let manifest = original
            .manifest_location
            .naming_scheme
            .manifest_path(&original.base, 2);
        original.object_store.inner.delete(&manifest).await.unwrap();

        let result = commit_keyed(&original, append(1, &[]), &["a"]).await;
        match result {
            Err(err @ Error::CommitConflict { .. }) => {
                assert!(
                    err.to_string().contains("versions [2] were cleaned up"),
                    "{err}"
                )
            }
            other => panic!("expected CommitConflict, got {other:?}"),
        }
        let unkeyed = CommitBuilder::new(original.clone())
            .execute(append(1, &[]))
            .await
            .unwrap();
        assert_eq!(unkeyed.manifest.version, 4);
    }

    /// The caller-named property and the reserved property interoperate:
    /// whichever API commits first, the second commit sees the keys.
    /// `fork_writer` commits only the caller-named property, as a writer
    /// that predates the reserved property does.
    #[rstest]
    #[case::property_then_keys(Writer::Property, Writer::Keys, true)]
    #[case::keys_then_property(Writer::Keys, Writer::Property, true)]
    #[case::property_then_property(Writer::Property, Writer::Property, true)]
    #[case::fork_writer_then_property(Writer::PropertyOnly, Writer::Property, true)]
    #[case::fork_writer_then_keys(Writer::PropertyOnly, Writer::Keys, false)]
    #[tokio::test]
    async fn test_caller_property_interop(
        #[case] first: Writer,
        #[case] second: Writer,
        #[case] is_duplicate: bool,
    ) {
        let original = dataset().await;
        let committed = first.commit(&original).await.unwrap();
        let persisted = committed.read_transaction().await.unwrap().unwrap();
        if first != Writer::Keys {
            // The caller reads its own property back from recent versions.
            assert_eq!(
                stored_keys(&persisted, CALLER_PROPERTY).as_deref(),
                Some(r#"["a","b"]"#)
            );
        }
        let result = second.commit(&original).await;
        if is_duplicate {
            let overlap = first
                .keys()
                .iter()
                .copied()
                .filter(|key| second.keys().contains(key))
                .collect::<Vec<_>>();
            assert_duplicate(result, 2, &overlap);
        } else {
            assert_eq!(result.unwrap().manifest.version, 3);
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Writer {
        /// `with_idempotency_keys(["a", "c"])`.
        Keys,
        /// Keys in the caller-named property, `with_idempotency_property`.
        Property,
        /// Keys in the caller-named property, committed without any
        /// idempotency configuration.
        PropertyOnly,
    }

    impl Writer {
        fn keys(self) -> &'static [&'static str] {
            match self {
                Self::Keys => &["a", "c"],
                Self::Property | Self::PropertyOnly => &["a", "b"],
            }
        }

        async fn commit(self, dataset: &Arc<Dataset>) -> Result<Dataset> {
            let builder = CommitBuilder::new(dataset.clone());
            match self {
                Self::Keys => {
                    builder
                        .with_idempotency_keys(["a", "c"])
                        .execute(append(1, &[]))
                        .await
                }
                Self::Property => {
                    builder
                        .with_idempotency_property(CALLER_PROPERTY)
                        .execute(append(1, &[(CALLER_PROPERTY, r#"["a","b"]"#)]))
                        .await
                }
                Self::PropertyOnly => {
                    builder
                        .execute(append(1, &[(CALLER_PROPERTY, r#"["a","b"]"#)]))
                        .await
                }
            }
        }
    }

    #[tokio::test]
    async fn test_caller_property_copies_keys() {
        let original = dataset().await;
        let committed = CommitBuilder::new(original.clone())
            .with_idempotency_property(CALLER_PROPERTY)
            .execute(append(1, &[(CALLER_PROPERTY, r#"["b","a"]"#)]))
            .await
            .unwrap();
        let persisted = committed.read_transaction().await.unwrap().unwrap();
        assert_eq!(
            stored_keys(&persisted, IDEMPOTENCY_KEYS_PROPERTY).as_deref(),
            Some(r#"["a","b"]"#)
        );
        assert_eq!(
            stored_keys(&persisted, CALLER_PROPERTY).as_deref(),
            Some(r#"["b","a"]"#)
        );
    }

    #[rstest]
    #[case::missing(&[], false, "is not set on the transaction")]
    #[case::malformed(&[(CALLER_PROPERTY, "a,b")], false, "is not a JSON array of strings")]
    #[case::with_keys(&[(CALLER_PROPERTY, r#"["a"]"#)], true, "mutually exclusive")]
    #[tokio::test]
    async fn test_caller_property_errors(
        #[case] properties: &[(&str, &str)],
        #[case] has_keys: bool,
        #[case] message: &str,
    ) {
        let original = dataset().await;
        let mut builder =
            CommitBuilder::new(original.clone()).with_idempotency_property(CALLER_PROPERTY);
        if has_keys {
            builder = builder.with_idempotency_keys(["a"]);
        }
        assert_invalid_input(builder.execute(append(1, properties)).await, message);
        assert_eq!(original.latest_version_id().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_detached_commit_rejects_keys() {
        let original = dataset().await;
        let result = CommitBuilder::new(original.clone())
            .with_detached(true)
            .with_idempotency_keys(["a"])
            .execute(append(1, &[]))
            .await;
        assert!(
            matches!(&result, Err(Error::NotSupported { .. })),
            "{result:?}"
        );
        // Unkeyed detached commits are unaffected.
        CommitBuilder::new(original)
            .with_detached(true)
            .execute(append(1, &[]))
            .await
            .unwrap();
    }

    /// Creating a dataset validates and records the keys; there is nothing
    /// before the first version to check them against.
    #[tokio::test]
    async fn test_new_dataset_records_keys() {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1]))])
            .unwrap();
        let reader = || RecordBatchIterator::new(vec![Ok(batch.clone())], schema.clone());

        // Rejected before any data file is written.
        let dir = TempStrDir::default();
        let params = WriteParams::default().with_idempotency_keys([""]);
        assert_invalid_input(
            Dataset::write(reader(), dir.as_str(), Some(params)).await,
            "non-empty",
        );
        assert!(!std::path::Path::new(dir.as_str()).join("data").exists());

        let params = WriteParams::default().with_idempotency_keys(["a"]);
        let created = Dataset::write(reader(), "memory://", Some(params))
            .await
            .unwrap();
        let persisted = created.read_transaction().await.unwrap().unwrap();
        assert_eq!(
            stored_keys(&persisted, IDEMPOTENCY_KEYS_PROPERTY).as_deref(),
            Some(r#"["a"]"#)
        );
    }

    /// A strict overwrite (no retries) never rebases: its manifest put only
    /// succeeds when nothing was committed after its read version, so a
    /// concurrent duplicate surfaces as a commit conflict.
    #[tokio::test]
    async fn test_strict_overwrite_with_keys() {
        let original = dataset().await;
        commit_keyed(&original, append(1, &[]), &["a"])
            .await
            .unwrap();
        let strict = |keys: &'static [&'static str]| {
            CommitBuilder::new(original.clone())
                .with_max_retries(0)
                .with_idempotency_keys(keys.iter().copied())
                .execute(overwrite(&original, 1))
        };
        assert_invalid_input(strict(&[""]).await, "non-empty");
        let result = strict(&["a"]).await;
        assert!(
            matches!(&result, Err(Error::CommitConflict { .. })),
            "{result:?}"
        );
    }

    #[tokio::test]
    async fn test_batch_commit_merges_keys() {
        let original = dataset().await;
        let batch = vec![
            append(1, &[(IDEMPOTENCY_KEYS_PROPERTY, r#"["a"]"#)]),
            append(1, &[]),
            append(1, &[(IDEMPOTENCY_KEYS_PROPERTY, r#"["b"]"#)]),
        ];
        let result = CommitBuilder::new(original.clone())
            .with_idempotency_keys(["c"])
            .execute_batch(batch)
            .await
            .unwrap();
        assert_eq!(
            stored_keys(&result.merged, IDEMPOTENCY_KEYS_PROPERTY).as_deref(),
            Some(r#"["a","b"]"#)
        );
        let persisted = result.dataset.read_transaction().await.unwrap().unwrap();
        assert_eq!(
            stored_keys(&persisted, IDEMPOTENCY_KEYS_PROPERTY).as_deref(),
            Some(r#"["a","b","c"]"#)
        );
        assert_duplicate(
            commit_keyed(&original, append(1, &[]), &["b"]).await,
            2,
            &["b"],
        );

        // An unkeyed batch commits no properties, as before.
        let unkeyed = CommitBuilder::new(Arc::new(result.dataset))
            .execute_batch(vec![append(2, &[])])
            .await
            .unwrap();
        assert_eq!(unkeyed.merged.transaction_properties, None);
    }

    #[rstest]
    #[case::shared_key(None, &[r#"["a","b"]"#, r#"["b"]"#], "shares the idempotency key 'b'")]
    #[case::caller_property_missing(Some(CALLER_PROPERTY), &[], "is not set on the transaction")]
    #[tokio::test]
    async fn test_batch_commit_errors(
        #[case] caller_property: Option<&str>,
        #[case] keys: &[&str],
        #[case] message: &str,
    ) {
        let original = dataset().await;
        let mut transactions = keys
            .iter()
            .map(|keys| append(1, &[(IDEMPOTENCY_KEYS_PROPERTY, keys)]))
            .collect::<Vec<_>>();
        if transactions.is_empty() {
            transactions.push(append(1, &[]));
        }
        let mut builder = CommitBuilder::new(original);
        if let Some(property) = caller_property {
            builder = builder.with_idempotency_property(property);
        }
        assert_invalid_input(
            builder.execute_batch(transactions).await.map(|_| ()),
            message,
        );
    }

    #[tokio::test]
    async fn test_batch_commit_with_caller_property() {
        let original = dataset().await;
        let batch = vec![
            append(1, &[(CALLER_PROPERTY, r#"["a"]"#)]),
            append(1, &[(CALLER_PROPERTY, r#"["b"]"#)]),
        ];
        let result = CommitBuilder::new(original.clone())
            .with_idempotency_property(CALLER_PROPERTY)
            .execute_batch(batch)
            .await
            .unwrap();
        let persisted = result.dataset.read_transaction().await.unwrap().unwrap();
        for name in [IDEMPOTENCY_KEYS_PROPERTY, CALLER_PROPERTY] {
            assert_eq!(
                stored_keys(&persisted, name).as_deref(),
                Some(r#"["a","b"]"#)
            );
        }
    }

    fn int_batch(values: Vec<i32>) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "x",
            DataType::Int32,
            false,
        )]));
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(values))]).unwrap()
    }

    /// Write `values` to `uri` with `keys`, committing at `read_version`
    /// through `handler`.
    async fn write_keyed(
        uri: &str,
        handler: &Arc<AmbiguousCommitHandler>,
        read_version: u64,
        values: Vec<i32>,
        keys: &[&str],
    ) -> Result<Dataset> {
        let dataset = Arc::new(
            Dataset::open(uri)
                .await
                .unwrap()
                .checkout_version(read_version)
                .await
                .unwrap(),
        );
        let params = WriteParams {
            mode: WriteMode::Append,
            commit_handler: Some(handler.clone()),
            ..Default::default()
        }
        .with_idempotency_keys(keys.iter().copied());
        crate::dataset::InsertBuilder::new(dataset)
            .with_params(&params)
            .execute(vec![int_batch(values)])
            .await
    }

    /// A keyed put that lands but reports a conflict is recognised as our
    /// own commit within the same call; a caller that retries a commit whose
    /// status was unknown gets [`Error::DuplicateTransaction`] naming its own
    /// version.
    #[rstest]
    #[case::landed_and_conflict(AmbiguousFailure::LandAndConflict)]
    #[case::status_unknown(AmbiguousFailure::LandAndError)]
    #[tokio::test]
    async fn test_ambiguous_commit(#[case] failure: AmbiguousFailure) {
        let tmp = TempStrDir::default();
        let uri = tmp.as_str();
        let handler = Arc::new(AmbiguousCommitHandler::default());
        let params = WriteParams {
            commit_handler: Some(handler.clone()),
            ..Default::default()
        };
        let batch = int_batch(vec![1, 2, 3]);
        let schema = batch.schema();
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            uri,
            Some(params),
        )
        .await
        .unwrap();

        handler.fail_next(failure);
        let is_unknown = failure == AmbiguousFailure::LandAndError;
        handler
            .fail_resolve
            .store(is_unknown, std::sync::atomic::Ordering::SeqCst);
        let first = write_keyed(uri, &handler, 1, vec![4, 5, 6], &["a"]).await;
        handler
            .fail_resolve
            .store(false, std::sync::atomic::Ordering::SeqCst);
        if is_unknown {
            let err = first.expect_err("verification was unavailable");
            assert!(err.is_commit_status_unknown(), "{err:?}");
            // The caller retries the work it could not confirm.
            let retry = write_keyed(uri, &handler, 1, vec![4, 5, 6], &["a"]).await;
            assert_duplicate(retry, 2, &["a"]);
        } else {
            assert_eq!(first.unwrap().manifest.version, 2);
        }
        let dataset = Dataset::open(uri).await.unwrap();
        assert_eq!(dataset.version().version, 2);
        assert_eq!(dataset.count_rows(None).await.unwrap(), 6);
    }
}
