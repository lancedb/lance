// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Key existence tracking for merge insert conflict detection.
//!
//! The implementation lives in [`lance_table::format::key_existence`] because the
//! filter is serialized into the transaction protobuf.

pub use lance_table::format::key_existence::*;
