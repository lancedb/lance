// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `Index`-trait adapter for the MemWAL system index.
//!
//! The data structures and table-format logic live in
//! [`lance_table::system_index::mem_wal`]; this module re-exports them and
//! implements the local [`Index`] trait for [`MemWalIndex`].

pub use lance_table::system_index::mem_wal::*;
