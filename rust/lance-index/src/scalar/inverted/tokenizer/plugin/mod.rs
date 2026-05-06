// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Tokenizer plugin support for dynamically loadable tokenizers.
//!
//! See `include/lance_tokenizer_plugin.h` for the C ABI contract and
//! `OwnedPluginFactory` in `loader.rs` for the Send/Sync and borrow-chain
//! design.
//!
//! # Plugin fault recovery
//!
//! A plugin callback that panics or aborts the host puts the process in
//! undefined-behavior territory; we make no attempt to resume. A panic that
//! escapes while an `OwnedPluginFactory` / `OwnedPluginTokenizerInstance`
//! mutex is held poisons that mutex. Fallible operations (`create_tokenizer`,
//! `create_stream`) surface poisoning as `Err` so callers can release the
//! per-instance `active_stream` reservation; `Drop` recovers the inner pointer
//! via `into_inner()` and still calls the plugin's `destroy_*` callback,
//! avoiding a double-panic that would escalate to `abort`.

pub mod ffi;
pub mod loader;
pub mod tokenizer;

pub use ffi::{CError, CStringRef, CToken, CTokenizerPlugin, PLUGIN_API_VERSION};
pub use loader::TokenizerPluginLibrary;
pub use tokenizer::PluginTokenizer;
