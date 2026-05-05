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
//! poison that mutex, and every subsequent call into the affected handle
//! panics through `expect("...mutex poisoned")`. This is intentional —
//! continuing to drive a plugin that just faulted would risk indexing
//! corrupted tokens or jumping to freed function pointers. Restart the
//! process to recover.

pub mod ffi;
pub mod loader;
pub mod tokenizer;

pub use ffi::{CError, CStringRef, CToken, CTokenizerPlugin, PLUGIN_API_VERSION};
pub use loader::TokenizerPluginLibrary;
pub use tokenizer::PluginTokenizer;
