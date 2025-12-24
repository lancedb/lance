// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#ifndef LANCE_TOKENIZER_PLUGIN_H
#define LANCE_TOKENIZER_PLUGIN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LANCE_TOKENIZER_PLUGIN_API_VERSION 1

/// The `text` pointer is valid only until the next call to `next_token()`
/// or `destroy_stream()`. Callers should copy the text if needed.
typedef struct LanceToken {
    /// Start and end byte offsets in the original text (UTF-8)
    uint32_t offset_from;
    uint32_t offset_to;

    /// Position of this token in the sequence (0-indexed)
    uint32_t position;
    uint32_t position_length;

    /// Pointer to the token text (null-terminated UTF-8)
    /// Valid until next `next_token()` or `destroy_stream()` call
    const char* text;
    uint32_t text_length;
} LanceToken;

typedef struct LanceTokenizerFactory LanceTokenizerFactory;
typedef struct LanceTokenizer LanceTokenizer;
typedef struct LanceTokenStream LanceTokenStream;

typedef struct LanceTokenizerPlugin {
    uint32_t (*api_version)(void);

    /// Create a tokenizer factory with the given JSON configuration.
    ///
    /// @param config_json JSON configuration string (UTF-8, null-terminated)
    /// @param config_len Length of config_json in bytes (not including null terminator)
    /// @return Factory handle, or NULL on error (call get_error for details)
    LanceTokenizerFactory* (*create_factory)(const char* config_json, uint32_t config_len);

    /// Destroy a factory and free its resources.
    ///
    /// @param factory Factory handle
    void (*destroy_factory)(LanceTokenizerFactory* factory);

    /// Create a tokenizer instance from the factory.
    /// Multiple tokenizers can be created from a single factory.
    ///
    /// @param factory Factory handle
    /// @return Tokenizer handle, or NULL on error (call get_error for details)
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
    /// @param text_length Length of text in bytes
    /// @return Stream handle, or NULL on error (call get_error for details)
    LanceTokenStream* (*create_stream)(LanceTokenizer* tokenizer, const char* text, uint32_t text_length);

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
