// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

/// Lance Tokenizer Plugin C API
///
/// This header defines the C ABI interface for implementing custom tokenizers
/// as dynamically loadable plugins. Plugins can be implemented in any language
/// that supports C ABI (Rust, C, C++, etc.).
///
/// ## Plugin Lifecycle
///
/// 1. Library is loaded, `lance_tokenizer_get_plugin()` is called to get plugin interface
/// 2. `create_factory()` is called with JSON configuration
/// 3. `create_tokenizer()` creates a tokenizer instance from the factory
/// 4. For each text to tokenize:
///    - `create_stream()` creates a token stream
///    - `next_token()` is called repeatedly until it returns 0
///    - `destroy_stream()` cleans up the stream
/// 5. `destroy_tokenizer()` cleans up the tokenizer
/// 6. `destroy_factory()` cleans up the factory
///
/// ## Error Handling
///
/// Functions that can fail return NULL (for pointers) or negative values (for int).
/// Call `get_error()` with the relevant handle to get the error message.
///
/// ## Example Configuration (JSON)
///
/// ```json
/// {
///   "mode": "search",
///   "dict_path": "/path/to/dictionary",
///   "user_dict": "/path/to/user_dict.txt"
/// }
/// ```

#ifndef LANCE_TOKENIZER_PLUGIN_H
#define LANCE_TOKENIZER_PLUGIN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Plugin API version - increment when making breaking changes
#define LANCE_TOKENIZER_PLUGIN_API_VERSION 1

/// Token information produced by the tokenizer
///
/// The `text` pointer is valid only until the next call to `next_token()`
/// or `destroy_stream()`. Callers should copy the text if needed.
typedef struct LanceToken {
    /// Start byte offset in the original text (UTF-8)
    uint32_t offset_from;

    /// End byte offset in the original text (UTF-8)
    uint32_t offset_to;

    /// Position of this token in the sequence (0-indexed)
    uint32_t position;

    /// Pointer to the token text (null-terminated UTF-8)
    /// Valid until next `next_token()` or `destroy_stream()` call
    const char* text;

    /// Length of the token text in bytes (not including null terminator)
    uint32_t text_len;

    /// Position length (usually 1, but can be > 1 for synonyms)
    uint32_t position_length;
} LanceToken;

/// Opaque factory handle - stores shared resources like dictionaries
typedef struct LanceTokenizerFactory LanceTokenizerFactory;

/// Opaque tokenizer handle - stateful tokenizer instance
typedef struct LanceTokenizer LanceTokenizer;

/// Opaque token stream handle - iterates over tokens for a single text
typedef struct LanceTokenStream LanceTokenStream;

/// Plugin interface - function pointers that plugins must implement
typedef struct LanceTokenizerPlugin {
    /// Returns the API version this plugin was built against.
    /// Must return LANCE_TOKENIZER_PLUGIN_API_VERSION.
    uint32_t (*api_version)(void);

    /// Create a tokenizer factory with the given JSON configuration.
    ///
    /// @param config_json JSON configuration string (UTF-8, null-terminated)
    /// @param config_len Length of config_json in bytes (not including null terminator)
    /// @return Factory handle, or NULL on error (call get_error for details)
    LanceTokenizerFactory* (*create_factory)(const char* config_json, uint32_t config_len);

    /// Destroy a factory and free its resources.
    ///
    /// @param factory Factory handle (may be NULL, which is a no-op)
    void (*destroy_factory)(LanceTokenizerFactory* factory);

    /// Create a tokenizer instance from the factory.
    /// Multiple tokenizers can be created from a single factory.
    ///
    /// @param factory Factory handle
    /// @return Tokenizer handle, or NULL on error
    LanceTokenizer* (*create_tokenizer)(LanceTokenizerFactory* factory);

    /// Destroy a tokenizer and free its resources.
    ///
    /// @param tokenizer Tokenizer handle (may be NULL, which is a no-op)
    void (*destroy_tokenizer)(LanceTokenizer* tokenizer);

    /// Create a token stream for the given text.
    /// The stream must be destroyed before creating another stream from the same tokenizer.
    ///
    /// @param tokenizer Tokenizer handle
    /// @param text Text to tokenize (UTF-8, not necessarily null-terminated)
    /// @param text_len Length of text in bytes
    /// @return Stream handle, or NULL on error
    LanceTokenStream* (*create_stream)(LanceTokenizer* tokenizer, const char* text, uint32_t text_len);

    /// Destroy a token stream and free its resources.
    ///
    /// @param stream Stream handle (may be NULL, which is a no-op)
    void (*destroy_stream)(LanceTokenStream* stream);

    /// Get the next token from the stream.
    ///
    /// @param stream Stream handle
    /// @param token Output parameter - filled with token data if a token is available
    /// @return 1 if a token was produced, 0 if no more tokens, negative on error
    int32_t (*next_token)(LanceTokenStream* stream, LanceToken* token);

    /// Get the last error message.
    ///
    /// @param factory Factory handle (can be NULL to get global/loading errors)
    /// @return Error message (null-terminated), or NULL if no error
    ///         The returned string is valid until the next error-generating call
    const char* (*get_error)(LanceTokenizerFactory* factory);

    /// Get the plugin name.
    ///
    /// @return Plugin name (null-terminated, statically allocated)
    const char* (*name)(void);

    /// Get the plugin version.
    ///
    /// @return Plugin version string (null-terminated, statically allocated)
    const char* (*version)(void);
} LanceTokenizerPlugin;

/// Entry point function type.
/// Plugins must export a function named `lance_tokenizer_get_plugin` with this signature.
typedef const LanceTokenizerPlugin* (*LanceTokenizerGetPluginFn)(void);

/// Entry point symbol name
#define LANCE_TOKENIZER_ENTRY_POINT "lance_tokenizer_get_plugin"

#ifdef __cplusplus
}
#endif

#endif // LANCE_TOKENIZER_PLUGIN_H
