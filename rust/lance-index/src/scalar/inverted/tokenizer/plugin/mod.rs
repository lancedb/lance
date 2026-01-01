// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Tokenizer plugin support for dynamically loadable tokenizers.
//!
//! The `TokenizerPluginLibrary` is `Send + Sync` and can be shared across threads.
//! However, `PluginFactory`, `PluginTokenizerInstance`, and `PluginTokenStream`
//! are not thread-safe and should be created per-thread as needed.

pub mod ffi;
pub mod loader;
pub mod tokenizer;

pub use ffi::{CError, CStringRef, CToken, CTokenizerPlugin, PLUGIN_API_VERSION};
pub use loader::TokenizerPluginLibrary;
pub use tokenizer::PluginTokenizer;
