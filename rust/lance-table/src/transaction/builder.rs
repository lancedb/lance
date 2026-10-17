// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The transaction itself: an operation plus the version it was based on.

use crate::transaction::Operation;
use lance_core::deepsize::DeepSizeOf;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// A change to a dataset that can be retried
///
/// This contains enough information to be able to build the next manifest,
/// given the current manifest.
#[derive(Debug, Clone, DeepSizeOf)]
pub struct Transaction {
    /// The version of the table this transaction is based off of. If this is
    /// the first transaction, this should be 0.
    pub read_version: u64,
    pub uuid: String,
    pub operation: Operation,
    pub tag: Option<String>,
    pub transaction_properties: Option<Arc<HashMap<String, String>>>,
    /// Whether this transaction's data writes embedded version 1 zone write
    /// seeds in the files they published. `Some` is a write-path fact recorded
    /// by the writer that held the seed collector, so the commit can set the
    /// paired `FLAG_ZONE_WRITE_SEEDS` bits without re-deriving the answer from
    /// index configuration, which a concurrent `DROP INDEX` can change between
    /// the write and the commit. `None` means the fact is unknown — the
    /// transaction was rebuilt from its serialized form or assembled outside
    /// the write path — and the commit must inspect the published files
    /// themselves. Deliberately not serialized: the files are the durable
    /// record, and a serialized claim could drift from them.
    pub wrote_v1_seeds: Option<bool>,
}

/// `wrote_v1_seeds` is write-path provenance, not part of the transaction's
/// identity: a transaction serialized and read back is still the same
/// transaction even though the serialized form does not carry the fact.
impl PartialEq for Transaction {
    fn eq(&self, other: &Self) -> bool {
        self.read_version == other.read_version
            && self.uuid == other.uuid
            && self.operation == other.operation
            && self.tag == other.tag
            && self.transaction_properties == other.transaction_properties
    }
}

/// Add TransactionBuilder for flexibly setting option without using `mut`
pub struct TransactionBuilder {
    read_version: u64,
    // uuid is optional for builder since it can autogenerate
    uuid: Option<String>,
    operation: Operation,
    tag: Option<String>,
    transaction_properties: Option<Arc<HashMap<String, String>>>,
    wrote_v1_seeds: Option<bool>,
}

impl TransactionBuilder {
    pub fn new(read_version: u64, operation: Operation) -> Self {
        Self {
            read_version,
            uuid: None,
            operation,
            tag: None,
            transaction_properties: None,
            wrote_v1_seeds: None,
        }
    }

    pub fn uuid(mut self, uuid: String) -> Self {
        self.uuid = Some(uuid);
        self
    }

    pub fn tag(mut self, tag: Option<String>) -> Self {
        self.tag = tag;
        self
    }

    pub fn transaction_properties(
        mut self,
        transaction_properties: Option<Arc<HashMap<String, String>>>,
    ) -> Self {
        self.transaction_properties = transaction_properties;
        self
    }

    /// Record whether the write that produced this transaction's files
    /// embedded version 1 zone write seeds. See [`Transaction::wrote_v1_seeds`].
    pub fn wrote_v1_seeds(mut self, wrote_v1_seeds: bool) -> Self {
        self.wrote_v1_seeds = Some(wrote_v1_seeds);
        self
    }

    pub fn build(self) -> Transaction {
        let uuid = self
            .uuid
            .unwrap_or_else(|| Uuid::new_v4().hyphenated().to_string());
        Transaction {
            read_version: self.read_version,
            uuid,
            operation: self.operation,
            tag: self.tag,
            transaction_properties: self.transaction_properties,
            wrote_v1_seeds: self.wrote_v1_seeds,
        }
    }
}

impl Transaction {
    pub fn new_from_version(read_version: u64, operation: Operation) -> Self {
        TransactionBuilder::new(read_version, operation).build()
    }

    pub fn new(read_version: u64, operation: Operation, tag: Option<String>) -> Self {
        TransactionBuilder::new(read_version, operation)
            .tag(tag)
            .build()
    }
}
