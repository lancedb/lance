// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Tokenizer plugin support for dynamically loadable tokenizers.
//!
//! This module provides support for loading tokenizers from shared libraries
//! at runtime, using a stable C ABI interface. This allows users to add custom
//! tokenizers without recompiling Lance.
//!
//! # Plugin Development
//!
//! To create a tokenizer plugin:
//!
//! 1. Include the C header file `include/lance_tokenizer_plugin.h`
//! 2. Implement all required functions in the `LanceTokenizerPlugin` struct
//! 3. Export the `lance_tokenizer_get_plugin` function
//! 4. Compile as a shared library (.so on Linux, .dylib on macOS, .dll on Windows)
//!
//! # Example Usage
//!
//! ```ignore
//! use lance_index::scalar::inverted::tokenizer::plugin::PluginTokenizer;
//!
//! // Load a plugin tokenizer
//! let tokenizer = PluginTokenizer::new(
//!     "/path/to/libmy_tokenizer.so",
//!     r#"{"mode": "search"}"#,
//! )?;
//!
//! // Use with inverted index
//! let params = InvertedIndexParams::default()
//!     .plugin_tokenizer(tokenizer);
//! ```
//!
//! # Thread Safety
//!
//! The `TokenizerPluginLibrary` is `Send + Sync` and can be shared across threads.
//! However, `PluginFactory`, `PluginTokenizerInstance`, and `PluginTokenStream`
//! are not thread-safe and should be created per-thread as needed.

pub mod ffi;
pub mod loader;
pub mod tokenizer;

pub use ffi::{CToken, CTokenizerPlugin, PLUGIN_API_VERSION};
pub use loader::TokenizerPluginLibrary;
pub use tokenizer::PluginTokenizer;
