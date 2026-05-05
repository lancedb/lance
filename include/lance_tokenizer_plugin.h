// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#ifndef LANCE_TOKENIZER_PLUGIN_H
#define LANCE_TOKENIZER_PLUGIN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LANCE_TOKENIZER_PLUGIN_API_VERSION 1

/* Reentrancy contract.
 *
 * Lance does not expose any callback API that a plugin can call back
 * into; the only API surface the plugin sees is the vtable in this
 * header, and Lance is the sole driver of those calls. The constraint
 * the plugin must respect is therefore narrower than "do not call
 * Lance":
 *
 *   - Lance holds an internal mutex for the duration of each callback
 *     against a given factory or tokenizer handle. A plugin callback
 *     that synchronously waits for another thread which is itself in a
 *     callback against the same handle will deadlock against that
 *     mutex.
 *   - In practice this means a callback must not block on another
 *     thread that is, directly or indirectly, dispatching against the
 *     same `LanceTokenizerFactory*` or `LanceTokenizer*`. It is fine for
 *     callbacks to use internal worker threads that do their own
 *     computation; the rule is purely about reentrant dispatch on the
 *     same handle.
 *
 * What the plugin can rely on:
 *
 *   - Lance never invokes callbacks for a single factory or tokenizer
 *     handle concurrently. The mutex above serializes them per handle.
 *   - Streams are exclusive: only one live `LanceTokenStream*` may exist
 *     per `LanceTokenizer*` at a time, so per-stream scratch state needs
 *     no internal locking.
 */

/// A reference to a UTF-8 string. This provides a zero-copy way to pass strings between Rust and C.
typedef struct LanceStringRef {
    const char* data;
    uint32_t length;
} LanceStringRef;

/// Error information returned by plugin functions.
/// The message is valid until the next call on the same object or until destruction.
typedef struct LanceError {
    LanceStringRef message;
} LanceError;

typedef struct LanceToken {
    /// Start and end byte offsets in the original text (UTF-8)
    uint32_t offset_from;
    uint32_t offset_to;

    /// Position of this token in the sequence (0-indexed)
    uint32_t position;
    uint32_t position_length;

    /// Token text.
    ///
    /// Plugins may return either:
    /// - A slice of the original input text (zero-copy)
    /// - A pointer to stream-owned scratch memory
    ///
    /// The pointer must remain valid until the next next_token() call
    /// or destroy_stream(), whichever comes first.
    ///
    /// Note: Lance copies the text immediately after each next_token() call,
    ///       so plugins may safely reuse internal buffers.
    LanceStringRef text;
} LanceToken;

typedef struct LanceTokenizerFactory LanceTokenizerFactory;
typedef struct LanceTokenizer LanceTokenizer;
typedef struct LanceTokenStream LanceTokenStream;

typedef struct LanceTokenizerPlugin {
    uint32_t (*api_version)(void);

    /// Create a tokenizer factory with the given configuration.
    ///
    /// @param config Configuration string in a plugin-defined format (UTF-8).
    ///               Lance passes this string unchanged from user configuration.
    ///               Plugins may use any format (JSON, YAML, custom DSL, etc.).
    /// @param error Output parameter for error details (may be NULL if not needed)
    /// @return Factory handle, or NULL on error
    LanceTokenizerFactory* (*create_factory)(LanceStringRef config, LanceError* error);

    /// Destroy a factory and free its resources.
    ///
    /// @param factory Factory handle (may be NULL, which is a no-op)
    void (*destroy_factory)(LanceTokenizerFactory* factory);

    /// Create a tokenizer instance from the factory.
    /// Multiple tokenizers can be created from a single factory.
    ///
    /// @param factory Factory handle
    /// @param error Output parameter for error details (may be NULL if not needed)
    /// @return Tokenizer handle, or NULL on error
    LanceTokenizer* (*create_tokenizer)(LanceTokenizerFactory* factory, LanceError* error);

    /// Destroy a tokenizer and free its resources.
    ///
    /// @param tokenizer Tokenizer handle (may be NULL, which is a no-op)
    void (*destroy_tokenizer)(LanceTokenizer* tokenizer);

    /// Create a token stream for the given text.
    /// The stream must be destroyed before creating another stream from the same tokenizer.
    ///
    /// @param tokenizer Tokenizer handle
    /// @param text Text to tokenize (UTF-8)
    /// @param error Output parameter for error details (may be NULL if not needed)
    /// @return Stream handle, or NULL on error
    LanceTokenStream* (*create_stream)(LanceTokenizer* tokenizer, LanceStringRef text, LanceError* error);

    /// Destroy a token stream and free its resources.
    ///
    /// @param stream Stream handle (may be NULL, which is a no-op)
    void (*destroy_stream)(LanceTokenStream* stream);

    /// Get the next token from the stream.
    ///
    /// @param stream Stream handle
    /// @param token Output parameter - filled with token data if a token is available
    /// @param error Output parameter for error details (may be NULL if not needed)
    /// @return 1 if a token was produced, 0 if no more tokens, negative on error
    int32_t (*next_token)(LanceTokenStream* stream, LanceToken* token, LanceError* error);

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
