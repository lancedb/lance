// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance v1 file implementation.
//!
//! This module is the canonical home of the v1 reader, writer, metadata, and
//! page-table grammar. V1 accepts footer versions `(0, 0)` through `(0, 2)` and
//! writes the `(0, 2)` identity used by [`writer::FileWriter`].

pub mod encoding;
pub mod format;
pub mod page_table;
pub mod reader;
pub mod writer;
