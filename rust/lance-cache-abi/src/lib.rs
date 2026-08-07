// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Experimental xabi-backed contract for dynamically loaded Lance cache
//! backends.
//!
//! This crate intentionally prototypes the cache backend ABI as a Rust async
//! trait and lets xabi generate the C-compatible vtable, future polling, typed
//! error transport, panic guards, and module lifetime checks.

pub mod xabi_contract;

pub use xabi_contract::*;
