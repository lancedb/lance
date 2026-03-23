// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! C/C++ bindings for the Lance columnar data format.
//!
//! This crate exposes Lance's functionality through a stable C-ABI with
//! opaque handle patterns and Arrow C Data Interface for zero-copy data exchange.
//!
//! # Safety
//!
//! All `extern "C"` functions in this crate follow the C FFI safety contract:
//! - Pointers must be valid and non-null (unless documented as nullable).
//! - Opaque handles must have been created by the corresponding `lance_*_open`
//!   or `lance_*_new` function and must not be used after `lance_*_close`/`lance_*_free`.
//! - The caller is responsible for freeing returned strings with `lance_free_string()`.
#![allow(clippy::missing_safety_doc)]

mod async_dispatcher;
mod batch;
mod dataset;
mod error;
mod helpers;
pub mod runtime;
mod scanner;

// Re-export all extern "C" symbols so they appear in the cdylib.
pub use batch::*;
pub use dataset::*;
pub use error::{
    lance_free_string, lance_last_error_code, lance_last_error_message, LanceErrorCode,
};
pub use scanner::*;
