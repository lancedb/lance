// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Tokenizer plugin support for dynamically loadable tokenizers.
//!
//! The `TokenizerPluginLibrary` is `Send + Sync` and can be shared across threads.
//! However, `PluginFactory`, `PluginTokenizerInstance`, and `PluginTokenStream`
//! are not thread-safe and should be created per-thread as needed.
//!
//! # Plugin fault recovery is not supported
//!
//! If a plugin callback panics or aborts the host (e.g., unwinds across the
//! FFI boundary, dereferences invalid memory, or calls `std::process::abort`),
//! the host process is in undefined-behavior territory and we make no attempt
//! to resume. Specifically: a panic that escapes a callback while the
//! `OwnedPluginFactory` / `OwnedPluginTokenizerInstance` mutex is held will
//! poison that mutex, and the *affected handle* is unusable thereafter.
//! Continuing to drive a plugin that just faulted would risk indexing
//! corrupted tokens or jumping to freed function pointers. Restart the
//! process — or, where the API permits, build a fresh `PluginTokenizer` —
//! to recover.
//!
//! ## How a poisoned mutex surfaces
//!
//! The exact way the host reports a poisoned lock depends on whether the
//! caller can react to a `Result`:
//!
//! - Fallible operations (`OwnedPluginFactory::create_tokenizer`,
//!   `OwnedPluginTokenizerInstance::create_stream`) return `Err` so the
//!   caller can release any reservation it holds (notably the per-instance
//!   `active_stream` slot) and decide whether to retry against a freshly
//!   built handle.
//! - `Drop` cannot return an error and must avoid double-panicking during
//!   unwind. Both `OwnedPluginFactory::drop` and
//!   `OwnedPluginTokenizerInstance::drop` recover the inner pointer from a
//!   poisoned guard via `into_inner()` and still call the plugin's
//!   `destroy_*` callback, so a single fault does not escalate into
//!   `abort` and the C-side resources are still released best-effort.

pub mod ffi;
pub mod loader;
pub mod tokenizer;

pub use ffi::{CError, CStringRef, CToken, CTokenizerPlugin, PLUGIN_API_VERSION};
pub use loader::TokenizerPluginLibrary;
pub use tokenizer::PluginTokenizer;
