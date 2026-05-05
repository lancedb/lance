// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::{Error, Result};
use serde::{Deserialize, Deserializer, Serialize};
use std::{env, path::PathBuf};

#[cfg(feature = "tokenizer-jieba")]
mod jieba;

pub mod document_tokenizer;
#[cfg(feature = "tokenizer-lindera")]
mod lindera;

#[cfg(feature = "tokenizer-plugin")]
pub mod plugin;

#[cfg(feature = "tokenizer-jieba")]
use jieba::JiebaTokenizerBuilder;

#[cfg(feature = "tokenizer-lindera")]
use lindera::LinderaTokenizerBuilder;

use crate::pbold;
use crate::scalar::inverted::tokenizer::document_tokenizer::{
    JsonTokenizer, LanceTokenizer, TextTokenizer,
};
pub use lance_tokenizer::Language;
use lance_tokenizer::{
    AsciiFoldingFilter, LowerCaser, NgramTokenizer, RawTokenizer, RemoveLongFilter,
    SimpleTokenizer, Stemmer, StopWordFilter, TextAnalyzer, TextAnalyzerBuilder,
    WhitespaceTokenizer,
};

/// Tokenizer configs
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InvertedIndexParams {
    /// lance tokenizer takes care of different data types, such as text, json, etc.
    /// - 'text': parsing input documents into tokens
    /// - 'json': parsing input json string into tokens
    /// - none: auto type inference
    pub(crate) lance_tokenizer: Option<String>,
    /// base tokenizer:
    /// - `simple`: splits tokens on whitespace and punctuation
    /// - `whitespace`: splits tokens on whitespace
    /// - `raw`: no tokenization
    /// - `lindera/*`: Lindera tokenizer
    /// - `jieba/*`: Jieba tokenizer
    ///
    /// `simple` is recommended for most cases and the default value
    pub(crate) base_tokenizer: String,

    /// language for stemming and stop words
    /// this is only used when `stem` or `remove_stop_words` is true
    pub(crate) language: Language,

    /// If true, store the position of the term in the document
    /// This can significantly increase the size of the index
    /// If false, only store the frequency of the term in the document
    /// Default is false
    #[serde(default)]
    pub(crate) with_position: bool,

    /// maximum token length
    /// - `None`: no limit
    /// - `Some(n)`: remove tokens longer than `n`
    pub(crate) max_token_length: Option<usize>,

    /// whether lower case tokens
    #[serde(default = "bool_true")]
    pub(crate) lower_case: bool,

    /// whether apply stemming
    #[serde(default = "bool_true")]
    pub(crate) stem: bool,

    /// whether remove stop words
    #[serde(default = "bool_true")]
    pub(crate) remove_stop_words: bool,

    /// use customized stop words.
    /// - `None`: use built-in stop words based on language
    /// - `Some(words)`: use customized stop words
    pub(crate) custom_stop_words: Option<Vec<String>>,

    /// ascii folding
    #[serde(default = "bool_true")]
    pub(crate) ascii_folding: bool,

    /// min ngram length
    #[serde(default = "default_min_ngram_length")]
    pub(crate) min_ngram_length: u32,

    /// max ngram length
    #[serde(default = "default_max_ngram_length")]
    pub(crate) max_ngram_length: u32,

    /// whether prefix only
    #[serde(default)]
    pub(crate) prefix_only: bool,

    /// Total memory limit in MiB for the build stage.
    ///
    /// This is split evenly across FTS workers at build time. By default Lance
    /// uses roughly `num_cpus / 2` workers, unless `LANCE_FTS_NUM_SHARDS` is set.
    /// If unset, each worker defaults to a 2 GiB build-time memory limit.
    ///
    /// This is a build-time only parameter and is not persisted with the index.
    #[serde(
        rename = "memory_limit",
        skip_serializing,
        default,
        alias = "worker_memory_limit_mb"
    )]
    pub(crate) memory_limit_mb: Option<u64>,

    /// Number of workers to use for FTS build.
    ///
    /// This is a build-time only parameter and is not persisted with the index.
    /// By default Lance uses roughly `num_cpus / 2` workers.
    /// The effective worker count is clamped to `[1, num_cpus - 2]`.
    #[serde(rename = "num_workers", skip_serializing, default)]
    pub(crate) num_workers: Option<usize>,

    /// Path to a tokenizer plugin shared library.
    /// When set, the plugin will be loaded and used instead of built-in tokenizers.
    /// Requires the `tokenizer-plugin` feature.
    ///
    /// The value is always absolute by the time it lands in this field: the
    /// `plugin()` builder absolutizes user input at construction time, and the
    /// serde deserializer absolutizes any relative path it reads back (e.g.,
    /// from JSON written by an older version of this code, or by an external
    /// producer). Serialization itself is a pure pass-through, so repeated
    /// `serde_json::to_string` calls on the same params are idempotent and do
    /// not depend on the current working directory.
    ///
    /// # Security
    ///
    /// Opening an index whose manifest carries this field will dlopen the
    /// referenced shared library and run its code inside the host process.
    /// Anyone with write access to the manifest can therefore achieve
    /// arbitrary native code execution. Only open indexes from sources you
    /// trust to the same level you trust the host process itself.
    ///
    /// # Portability
    ///
    /// The persisted path is the writer's absolute path. An index built on
    /// one machine will not resolve the plugin on another unless the same
    /// absolute path exists there too. Plugin tokenizers are currently
    /// intended for indexes that stay on a single host.
    #[serde(default, deserialize_with = "deserialize_plugin_path")]
    pub(crate) tokenizer_plugin_library: Option<String>,

    /// Configuration string for the tokenizer plugin (format defined by plugin).
    #[serde(default)]
    pub(crate) tokenizer_plugin_config: Option<String>,
}

impl TryFrom<&InvertedIndexParams> for pbold::InvertedIndexDetails {
    type Error = Error;

    fn try_from(params: &InvertedIndexParams) -> Result<Self> {
        Ok(Self {
            base_tokenizer: Some(params.base_tokenizer.clone()),
            language: serde_json::to_string(&params.language)?,
            with_position: params.with_position,
            max_token_length: params.max_token_length.map(|l| l as u32),
            lower_case: params.lower_case,
            stem: params.stem,
            remove_stop_words: params.remove_stop_words,
            ascii_folding: params.ascii_folding,
            min_ngram_length: params.min_ngram_length,
            max_ngram_length: params.max_ngram_length,
            prefix_only: params.prefix_only,
            // The value is already absolute at this point — the `plugin()`
            // builder and the serde deserializer both absolutize on the way
            // in, so we just clone through.
            tokenizer_plugin_library: params.tokenizer_plugin_library.clone(),
            tokenizer_plugin_config: params.tokenizer_plugin_config.clone(),
        })
    }
}

impl TryFrom<&pbold::InvertedIndexDetails> for InvertedIndexParams {
    type Error = Error;

    fn try_from(details: &pbold::InvertedIndexDetails) -> Result<Self> {
        let defaults = Self::default();
        Ok(Self {
            lance_tokenizer: defaults.lance_tokenizer,
            base_tokenizer: details
                .base_tokenizer
                .as_ref()
                .cloned()
                .unwrap_or(defaults.base_tokenizer),
            language: serde_json::from_str(details.language.as_str())?,
            with_position: details.with_position,
            max_token_length: details.max_token_length.map(|l| l as usize),
            lower_case: details.lower_case,
            stem: details.stem,
            remove_stop_words: details.remove_stop_words,
            custom_stop_words: defaults.custom_stop_words,
            ascii_folding: details.ascii_folding,
            min_ngram_length: details.min_ngram_length,
            max_ngram_length: details.max_ngram_length,
            prefix_only: details.prefix_only,
            memory_limit_mb: defaults.memory_limit_mb,
            num_workers: defaults.num_workers,
            tokenizer_plugin_library: details.tokenizer_plugin_library.clone(),
            tokenizer_plugin_config: details.tokenizer_plugin_config.clone(),
        })
    }
}

fn bool_true() -> bool {
    true
}

fn default_min_ngram_length() -> u32 {
    3
}

fn default_max_ngram_length() -> u32 {
    3
}

impl Default for InvertedIndexParams {
    fn default() -> Self {
        Self::new("simple".to_owned(), Language::English)
    }
}

impl InvertedIndexParams {
    /// Create a new `InvertedIndexParams` with the given base tokenizer and language.
    ///
    /// The `base_tokenizer` can be one of the following:
    /// - `simple`: splits tokens on whitespace and punctuation, default
    /// - `whitespace`: splits tokens on whitespace
    /// - `raw`: no tokenization
    /// - `ngram`: N-Gram tokenizer
    /// - `lindera/*`: Lindera tokenizer
    /// - `jieba/*`: Jieba tokenizer
    ///
    /// The `language` is used for stemming and removing stop words,
    /// this is not used for `lindera/*` and `jieba/*` tokenizers.
    /// Default to `English`.
    pub fn new(base_tokenizer: String, language: Language) -> Self {
        Self {
            lance_tokenizer: None,
            base_tokenizer,
            language,
            with_position: false,
            max_token_length: Some(40),
            lower_case: true,
            stem: true,
            remove_stop_words: true,
            custom_stop_words: None,
            ascii_folding: true,
            min_ngram_length: default_min_ngram_length(),
            max_ngram_length: default_max_ngram_length(),
            prefix_only: false,
            memory_limit_mb: None,
            num_workers: None,
            tokenizer_plugin_library: None,
            tokenizer_plugin_config: None,
        }
    }

    pub fn lance_tokenizer(mut self, lance_tokenizer: String) -> Self {
        self.lance_tokenizer = Some(lance_tokenizer);
        self
    }

    pub fn base_tokenizer(mut self, base_tokenizer: String) -> Self {
        self.base_tokenizer = base_tokenizer;
        self
    }

    pub fn language(mut self, language: &str) -> Result<Self> {
        // need to convert to valid JSON string
        let language = serde_json::from_str(format!("\"{}\"", language).as_str())?;
        self.language = language;
        Ok(self)
    }

    /// Set whether to store the position of the term in the document.
    /// This can significantly increase the size of the index.
    /// If false, only store the frequency of the term in the document.
    /// This doesn't work with `ngram` tokenizer.
    /// Default to `false`.
    pub fn with_position(mut self, with_position: bool) -> Self {
        self.with_position = with_position;
        self
    }

    /// Get whether positions are stored in this index.
    pub fn has_positions(&self) -> bool {
        self.with_position
    }

    pub fn max_token_length(mut self, max_token_length: Option<usize>) -> Self {
        self.max_token_length = max_token_length;
        self
    }

    pub fn lower_case(mut self, lower_case: bool) -> Self {
        self.lower_case = lower_case;
        self
    }

    pub fn stem(mut self, stem: bool) -> Self {
        self.stem = stem;
        self
    }

    pub fn remove_stop_words(mut self, remove_stop_words: bool) -> Self {
        self.remove_stop_words = remove_stop_words;
        self
    }

    pub fn custom_stop_words(mut self, custom_stop_words: Option<Vec<String>>) -> Self {
        self.custom_stop_words = custom_stop_words;
        self
    }

    pub fn ascii_folding(mut self, ascii_folding: bool) -> Self {
        self.ascii_folding = ascii_folding;
        self
    }

    /// Set the minimum N-Gram length, only works when `base_tokenizer` is `ngram`.
    /// Must be greater than 0 and not greater than `max_ngram_length`.
    /// Default to 3.
    pub fn ngram_min_length(mut self, min_length: u32) -> Self {
        self.min_ngram_length = min_length;
        self
    }

    /// Set the maximum N-Gram length, only works when `base_tokenizer` is `ngram`.
    /// Must be greater than 0 and not less than `min_ngram_length`.
    /// Default to 3.
    pub fn ngram_max_length(mut self, max_length: u32) -> Self {
        self.max_ngram_length = max_length;
        self
    }

    /// Set whether only prefix N-Gram is generated, only works when `base_tokenizer` is `ngram`.
    /// Default to `false`.
    pub fn ngram_prefix_only(mut self, prefix_only: bool) -> Self {
        self.prefix_only = prefix_only;
        self
    }

    pub fn memory_limit_mb(mut self, memory_limit_mb: u64) -> Self {
        self.memory_limit_mb = Some(memory_limit_mb);
        self
    }

    /// Set the number of workers to use for this build.
    ///
    /// By default Lance uses roughly `num_cpus / 2` workers.
    /// The effective worker count is clamped to `[1, num_cpus - 2]`.
    pub fn num_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = Some(num_workers);
        self
    }

    /// Serialize params for the build/training path, including build-only fields.
    pub fn to_training_json(&self) -> serde_json::Result<serde_json::Value> {
        let mut value = serde_json::to_value(self)?;
        let object = value
            .as_object_mut()
            .expect("inverted index params should serialize to a JSON object");
        if let Some(memory_limit_mb) = self.memory_limit_mb {
            object.insert(
                "memory_limit".to_string(),
                serde_json::Value::from(memory_limit_mb),
            );
        }
        if let Some(num_workers) = self.num_workers {
            object.insert(
                "num_workers".to_string(),
                serde_json::Value::from(num_workers),
            );
        }
        Ok(value)
    }

    /// Set a tokenizer plugin to use.
    /// `library_path` is path to the plugin shared library (.so, .dylib, or .dll)
    /// `config` is the configuration string for the plugin (format defined by plugin).
    /// This is required because the "empty" representation varies by format
    /// (e.g., "{}" for JSON, "" for plain text).
    ///
    /// Relative paths are resolved against the current working directory at
    /// the time this method is called and stored as absolute. Otherwise the
    /// persisted path would be re-interpreted against the CWD of whichever
    /// process later reopens the index, which can fail to find the plugin or,
    /// worse, silently load a different file with the same relative name.
    ///
    /// # Filter chain
    ///
    /// Tokens emitted by the plugin flow through the same post-processing
    /// chain as the built-in tokenizers. Defaults applied to plugin output:
    ///
    /// - `max_token_length = Some(40)` — drops tokens longer than 40 bytes
    /// - `lower_case = true` — ASCII-lowercases every token
    /// - `stem = true` — applies the configured language stemmer
    /// - `remove_stop_words = true` — drops language-specific stop words
    /// - `ascii_folding = true` — folds Latin diacritics to ASCII
    ///
    /// If the plugin already case-normalizes, stems, or strips diacritics on
    /// its side, set the corresponding option to `false` (e.g.
    /// `params.lower_case(false).stem(false).ascii_folding(false)`); otherwise
    /// the filter chain will apply on top of what the plugin produced and may
    /// surprise users who expect the plugin to be the sole source of truth.
    ///
    /// # Security
    ///
    /// The shared library at `library_path` is loaded with `dlopen` (or the
    /// platform equivalent) and its code runs inside the host process.
    /// Reopening an index whose manifest carries a plugin path will
    /// transparently do the same load. Anyone with write access to the
    /// manifest — a misconfigured object store, a shared filesystem, an
    /// untrusted index publisher — can therefore achieve arbitrary native
    /// code execution. Only enable plugin tokenizers, and only open indexes
    /// that use them, when you trust the source as much as the host process.
    ///
    /// # Portability
    ///
    /// `library_path` is absolutized against the current working directory
    /// and stored verbatim in the index manifest. An index built on one
    /// machine will not resolve the plugin on another unless the same
    /// absolute path exists there. Plugin tokenizers are currently intended
    /// for indexes that stay on a single host; portable basename + search
    /// directory resolution is a future extension.
    ///
    /// # Bindings
    ///
    /// The plugin tokenizer is exposed only through the Rust API today. The
    /// Python (`pyo3`) and Java (JNI) bindings have not yet been extended
    /// with `tokenizer_plugin_library` / `tokenizer_plugin_config`
    /// parameters, so plugin tokenizers cannot be configured from those
    /// languages. The conservative path is intentional: `dlopen` plus
    /// untrusted index manifests (see the security section) means we want
    /// the rollout to start with the smallest API surface and expand once
    /// the security model is settled.
    pub fn plugin(mut self, library_path: String, config: String) -> Self {
        self.base_tokenizer = "plugin".to_string();
        self.tokenizer_plugin_library = Some(absolutize_plugin_path(library_path));
        self.tokenizer_plugin_config = Some(config);
        self
    }

    pub fn build(&self) -> Result<Box<dyn LanceTokenizer>> {
        self.validate_plugin_consistency()?;
        let mut builder = self.build_base_tokenizer()?;

        if let Some(max_token_length) = self.max_token_length {
            builder = builder.filter_dynamic(RemoveLongFilter::limit(max_token_length));
        }
        if self.lower_case {
            builder = builder.filter_dynamic(LowerCaser);
        }
        if self.stem {
            builder = builder.filter_dynamic(Stemmer::new(self.language));
        }
        if self.remove_stop_words {
            let stop_word_filter = match &self.custom_stop_words {
                Some(words) => StopWordFilter::remove(words.iter().cloned()),
                None => StopWordFilter::new(self.language).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "removing stop words for language {:?} is not supported yet",
                        self.language
                    ))
                })?,
            };
            builder = builder.filter_dynamic(stop_word_filter);
        }
        if self.ascii_folding {
            builder = builder.filter_dynamic(AsciiFoldingFilter);
        }
        let tokenizer = builder.build();

        match self.lance_tokenizer.as_deref() {
            Some("text") | None => Ok(Box::new(TextTokenizer::new(tokenizer))),
            Some("json") => Ok(Box::new(JsonTokenizer::new(tokenizer))),
            Some(other) => Err(Error::invalid_input(format!(
                "unknown lance tokenizer {}",
                other
            ))),
        }
    }

    /// Reject parameter sets that mix the plugin fields with a non-`"plugin"`
    /// base tokenizer.
    ///
    /// Without this check, `tokenizer_plugin_library` / `tokenizer_plugin_config`
    /// arriving via JSON or proto deserialization would be silently ignored
    /// whenever `base_tokenizer` was left at its default `"simple"`. The user
    /// would believe they configured a plugin tokenizer (the metadata even
    /// keeps the path), while the index was actually built with the built-in
    /// `simple` tokenizer — and search results would silently use a different
    /// tokenizer than indexing did. This guards both the build-side and the
    /// post-deserialization paths in one place.
    fn validate_plugin_consistency(&self) -> Result<()> {
        let has_plugin_fields =
            self.tokenizer_plugin_library.is_some() || self.tokenizer_plugin_config.is_some();
        if has_plugin_fields && self.base_tokenizer != "plugin" {
            return Err(Error::invalid_input(format!(
                "tokenizer_plugin_library / tokenizer_plugin_config are set \
                 but base_tokenizer is {:?}; expected \"plugin\". Use \
                 InvertedIndexParams::plugin(library, config) to configure a \
                 plugin tokenizer, or unset the plugin fields if you intended \
                 to use the {:?} tokenizer.",
                self.base_tokenizer, self.base_tokenizer,
            )));
        }
        Ok(())
    }

    fn build_base_tokenizer(&self) -> Result<TextAnalyzerBuilder> {
        match self.base_tokenizer.as_str() {
            "simple" => Ok(TextAnalyzer::builder(SimpleTokenizer::default()).dynamic()),
            "whitespace" => Ok(TextAnalyzer::builder(WhitespaceTokenizer::default()).dynamic()),
            "raw" => Ok(TextAnalyzer::builder(RawTokenizer::default()).dynamic()),
            "ngram" => {
                let tokenizer = NgramTokenizer::new(
                    self.min_ngram_length as usize,
                    self.max_ngram_length as usize,
                    self.prefix_only,
                )
                .map_err(|e| Error::invalid_input(e.to_string()))?;
                Ok(TextAnalyzer::builder(tokenizer).dynamic())
            }
            #[cfg(feature = "tokenizer-lindera")]
            s if s.starts_with("lindera/") => {
                let Some(home) = language_model_home() else {
                    return Err(Error::invalid_input(format!(
                        "unknown base tokenizer {}",
                        self.base_tokenizer
                    )));
                };
                lindera::LinderaBuilder::load(&home.join(s))?.build()
            }
            #[cfg(feature = "tokenizer-jieba")]
            s if s.starts_with("jieba/") || s == "jieba" => {
                let s = if s == "jieba" { "jieba/default" } else { s };
                let Some(home) = language_model_home() else {
                    return Err(Error::invalid_input(format!(
                        "unknown base tokenizer {}",
                        self.base_tokenizer
                    )));
                };
                jieba::JiebaBuilder::load(&home.join(s))?.build()
            }
            "plugin" => self.build_plugin_tokenizer(),
            _ => Err(Error::invalid_input(format!(
                "unknown base tokenizer {}",
                self.base_tokenizer
            ))),
        }
    }

    fn build_plugin_tokenizer(&self) -> Result<TextAnalyzerBuilder> {
        #[cfg(feature = "tokenizer-plugin")]
        {
            use plugin::PluginTokenizer;

            let plugin_path = self.tokenizer_plugin_library.as_ref().ok_or_else(|| {
                Error::invalid_input(
                    "base_tokenizer is 'plugin' but tokenizer_plugin_library is not set",
                )
            })?;

            // Reject the empty string up front rather than letting it
            // propagate into `libloading::Library::new`, which surfaces it as
            // a relatively opaque platform-specific error (e.g.
            // `dlopen("")` reports the executable's own image on macOS, and
            // `LoadLibrary("")` returns ERROR_MOD_NOT_FOUND on Windows).
            if plugin_path.is_empty() {
                return Err(Error::invalid_input(
                    "tokenizer_plugin_library is set to an empty string; \
                     provide an absolute path to the plugin shared library",
                ));
            }

            let config = self.tokenizer_plugin_config.as_ref().ok_or_else(|| {
                Error::invalid_input(
                    "base_tokenizer is 'plugin' but tokenizer_plugin_config is not set.",
                )
            })?;

            // No `Path::exists()` precheck: `libloading::Library::new` (wrapped
            // by `TokenizerPluginLibrary::load`) already produces a precise
            // error for the underlying cause — file not found, missing
            // dependent shared library, permission denied, missing symbols,
            // ABI version mismatch — and an extra existence check would only
            // flatten those into a generic "file not found" while introducing
            // a TOCTOU window with `Library::new`.
            let tokenizer = PluginTokenizer::new(plugin_path, config)?;
            Ok(TextAnalyzer::builder(tokenizer).dynamic())
        }

        #[cfg(not(feature = "tokenizer-plugin"))]
        Err(Error::invalid_input(
            "tokenizer-plugin feature is not enabled, cannot use plugin tokenizers",
        ))
    }
}

/// Deserializer hook that absolutizes a plugin library path on read.
///
/// Callers may construct `InvertedIndexParams` from JSON produced by another
/// process (or by an older version of this code) where the path is still
/// relative; resolving here keeps the in-memory value consistent with what the
/// `plugin()` builder would produce, regardless of the producer's CWD.
///
/// Serialization is intentionally a pass-through: absolutization happens once,
/// at the input boundary, so repeated `serde_json::to_string` calls on the
/// same params are idempotent and do not drift if the process `chdir`s
/// between calls.
///
/// Non-UTF-8 path components survive `Path::to_string_lossy` as U+FFFD
/// replacement characters; in that case `Library::new` will fail with "file
/// not found" when the plugin is loaded, which surfaces a clear error at the
/// boundary where it can be reported to the user.
fn deserialize_plugin_path<'de, D: Deserializer<'de>>(
    deser: D,
) -> std::result::Result<Option<String>, D::Error> {
    Ok(Option::<String>::deserialize(deser)?.map(absolutize_plugin_path))
}

/// Convert a (possibly relative) plugin library path to absolute. This must
/// happen *before* the path is persisted into the index manifest, because the
/// process that reopens the index may have a different CWD than the one that
/// created it; resolving a relative path lazily would either fail to find the
/// plugin or load a wrong file with the same relative name.
fn absolutize_plugin_path(library_path: String) -> String {
    // Pass an empty string through unchanged. `Path::new("")` is otherwise
    // treated as a relative path, so prepending the CWD would turn `""` into
    // a directory path and quietly defeat the empty-string check in
    // `build_plugin_tokenizer`. The downstream check then surfaces a
    // descriptive "tokenizer_plugin_library is set to an empty string"
    // error instead of a confusing "is a directory" / dlopen failure.
    if library_path.is_empty() {
        return library_path;
    }
    let path = std::path::Path::new(&library_path);
    if path.is_absolute() {
        return library_path;
    }
    // Best effort: prepend CWD. If `current_dir()` fails (extremely rare —
    // e.g. the directory was deleted out from under the process), fall back
    // to the original string so the caller still sees a useful error from
    // the eager validation in `PluginTokenizer::new`.
    match env::current_dir() {
        Ok(cwd) => cwd.join(path).to_string_lossy().into_owned(),
        Err(_) => library_path,
    }
}

pub const LANCE_LANGUAGE_MODEL_HOME_ENV_KEY: &str = "LANCE_LANGUAGE_MODEL_HOME";

pub const LANCE_LANGUAGE_MODEL_DEFAULT_DIRECTORY: &str = "lance/language_models";

pub fn language_model_home() -> Option<PathBuf> {
    match env::var(LANCE_LANGUAGE_MODEL_HOME_ENV_KEY) {
        Ok(p) => Some(PathBuf::from(p)),
        Err(_) => dirs::data_local_dir().map(|p| p.join(LANCE_LANGUAGE_MODEL_DEFAULT_DIRECTORY)),
    }
}

#[cfg(test)]
mod tests {
    use super::InvertedIndexParams;
    use serial_test::serial;

    #[test]
    fn test_build_only_fields_are_not_serialized() {
        let params = InvertedIndexParams::default()
            .memory_limit_mb(4096)
            .num_workers(7);
        let json = serde_json::to_value(&params).unwrap();
        assert!(json.get("memory_limit").is_none());
        assert!(json.get("num_workers").is_none());
    }

    #[test]
    fn test_memory_limit_serde_accepts_legacy_worker_field_name() {
        let mut json = serde_json::to_value(InvertedIndexParams::default()).unwrap();
        let obj = json.as_object_mut().unwrap();
        obj.remove("memory_limit");
        obj.insert(
            "worker_memory_limit_mb".to_string(),
            serde_json::Value::from(2048),
        );
        let params: InvertedIndexParams = serde_json::from_value(json).unwrap();
        assert_eq!(params.memory_limit_mb, Some(2048));
    }

    #[test]
    fn test_build_only_fields_deserialize_from_public_names() {
        let mut json = serde_json::to_value(InvertedIndexParams::default()).unwrap();
        let obj = json.as_object_mut().unwrap();
        obj.insert("memory_limit".to_string(), serde_json::Value::from(4096));
        obj.insert("num_workers".to_string(), serde_json::Value::from(3));

        let params: InvertedIndexParams = serde_json::from_value(json).unwrap();
        assert_eq!(params.memory_limit_mb, Some(4096));
        assert_eq!(params.num_workers, Some(3));
    }

    #[test]
    fn test_training_json_serializes_build_only_fields() {
        let params = InvertedIndexParams::default()
            .memory_limit_mb(4096)
            .num_workers(3);
        let json = params.to_training_json().unwrap();
        assert_eq!(
            json.get("memory_limit"),
            Some(&serde_json::Value::from(4096))
        );
        assert_eq!(json.get("num_workers"), Some(&serde_json::Value::from(3)));
    }

    /// `write_metadata` persists `InvertedIndexParams` through `serde_json`,
    /// so the JSON it produces must already carry an absolute plugin library
    /// path; otherwise an index built with a relative path becomes unusable
    /// when reopened from a different CWD. Absolutization happens at the
    /// `plugin()` builder boundary, so serialization itself only needs to
    /// faithfully echo what the struct holds.
    #[test]
    fn test_plugin_path_is_absolute_in_serialized_json() {
        let params = InvertedIndexParams::default()
            .plugin("relative/dir/lib.so".to_string(), "{}".to_string());
        let json = serde_json::to_value(&params).unwrap();
        let serialized_path = json
            .get("tokenizer_plugin_library")
            .and_then(|v| v.as_str())
            .expect("serialized JSON should contain plugin path");
        assert!(
            std::path::Path::new(serialized_path).is_absolute(),
            "JSON-serialized plugin path must be absolute, got {:?}",
            serialized_path
        );
    }

    /// Serialization must be a pure pass-through of the in-memory value:
    /// repeating it on the same params (without rebuilding) must produce
    /// byte-identical JSON, even if the process CWD has changed between
    /// calls. This guards against regressing to the previous design where
    /// `serialize` re-evaluated `current_dir()` and could write a different
    /// absolute path on each call.
    ///
    /// `chdir` is process-wide, so this test must run serially with any other
    /// test that depends on CWD (notably the builder-side absolutization
    /// tests).
    #[test]
    #[serial(plugin_cwd)]
    fn test_plugin_path_serialize_is_idempotent_across_chdir() {
        struct CwdGuard(std::path::PathBuf);
        impl Drop for CwdGuard {
            fn drop(&mut self) {
                let _ = std::env::set_current_dir(&self.0);
            }
        }

        let _guard = CwdGuard(std::env::current_dir().unwrap());
        let tmp = tempfile::tempdir().unwrap();

        let params = InvertedIndexParams::default()
            .plugin("relative/dir/lib.so".to_string(), "{}".to_string());
        let first = serde_json::to_string(&params).unwrap();

        // chdir somewhere else, then serialize again. The output must not
        // change — the absolute path was baked in at builder time.
        std::env::set_current_dir(tmp.path()).unwrap();
        let second = serde_json::to_string(&params).unwrap();

        assert_eq!(
            first, second,
            "serialize must not re-evaluate current_dir; output drifted across chdir"
        );
    }

    /// Defense in depth: if an external producer hands us JSON with a
    /// relative plugin path (or an older version of this code wrote one),
    /// the deserializer must normalize it on read so downstream code never
    /// sees a CWD-dependent value.
    #[test]
    fn test_plugin_path_absolutized_on_json_deserialize() {
        // Start from a serialized default and inject a relative plugin
        // path, simulating JSON from an external producer.
        let mut json = serde_json::to_value(InvertedIndexParams::default()).unwrap();
        let obj = json.as_object_mut().unwrap();
        obj.insert(
            "tokenizer_plugin_library".to_string(),
            serde_json::Value::from("some/relative/lib.so"),
        );
        let params: InvertedIndexParams = serde_json::from_value(json).unwrap();
        let stored = params
            .tokenizer_plugin_library
            .as_ref()
            .expect("plugin library path should be set");
        assert!(
            std::path::Path::new(stored).is_absolute(),
            "deserialized plugin path must be absolute, got {:?}",
            stored
        );
    }

    /// `tokenizer_plugin_library` / `tokenizer_plugin_config` arriving via
    /// JSON or proto deserialization must not be silently ignored when
    /// `base_tokenizer` is left at its default. Without validation the
    /// resulting index would be built with the built-in `simple` tokenizer
    /// while still persisting the plugin path in metadata, leading to a
    /// silent mismatch between indexing and search tokenizers.
    #[test]
    fn test_build_rejects_plugin_fields_without_plugin_base_tokenizer() {
        let mut json = serde_json::to_value(InvertedIndexParams::default()).unwrap();
        let obj = json.as_object_mut().unwrap();
        obj.insert(
            "tokenizer_plugin_library".to_string(),
            serde_json::Value::from("/tmp/lib.so"),
        );
        obj.insert(
            "tokenizer_plugin_config".to_string(),
            serde_json::Value::from("{}"),
        );
        // Note: `base_tokenizer` is left at the default ("simple").
        let params: InvertedIndexParams = serde_json::from_value(json).unwrap();
        assert_eq!(params.base_tokenizer, "simple");

        // `Box<dyn LanceTokenizer>` doesn't implement `Debug`, so we cannot
        // use `expect_err` directly.
        match params.build() {
            Ok(_) => panic!("build must reject plugin fields with non-plugin base tokenizer"),
            Err(err) => {
                let msg = err.to_string();
                assert!(
                    msg.contains("base_tokenizer") && msg.contains("plugin"),
                    "error must name the conflicting fields, got: {}",
                    msg
                );
            }
        }
    }

    /// Sanity check the inverse: a default params object without plugin
    /// fields must still build cleanly through `simple` and not be
    /// blocked by the new validation.
    #[test]
    fn test_build_default_params_still_succeeds() {
        let params = InvertedIndexParams::default();
        params
            .build()
            .expect("default params must still build a simple tokenizer");
    }

    /// An empty `tokenizer_plugin_library` would otherwise descend into
    /// `libloading::Library::new("")`, which surfaces a relatively
    /// opaque platform-specific error (the executable's own image, or
    /// `ERROR_MOD_NOT_FOUND` on Windows). The boundary check should turn
    /// that into a self-explanatory error before we attempt to load.
    #[cfg(feature = "tokenizer-plugin")]
    #[test]
    fn test_build_rejects_empty_plugin_library_path() {
        let params = InvertedIndexParams {
            base_tokenizer: "plugin".to_string(),
            tokenizer_plugin_library: Some(String::new()),
            tokenizer_plugin_config: Some("{}".to_string()),
            ..InvertedIndexParams::default()
        };

        match params.build() {
            Ok(_) => panic!("build must reject an empty plugin library path"),
            Err(err) => {
                let msg = err.to_string();
                assert!(
                    msg.contains("empty string"),
                    "error must call out the empty path, got: {}",
                    msg
                );
            }
        }
    }

    /// `Path::new("")` is treated as a relative path, so a naive
    /// "prepend CWD if not absolute" implementation would turn an empty
    /// path into the current directory and silently defeat the
    /// downstream empty-string check. `absolutize_plugin_path` and the
    /// `plugin()` builder must keep an empty string empty so the
    /// `build_plugin_tokenizer` boundary check fires with a clear error.
    #[test]
    fn test_plugin_builder_keeps_empty_path_empty() {
        let params = InvertedIndexParams::default().plugin(String::new(), "{}".to_string());
        assert_eq!(
            params.tokenizer_plugin_library.as_deref(),
            Some(""),
            "plugin builder must not absolutize an empty library path \
             into the CWD"
        );
    }

    /// Same boundary, reached via JSON deserialization. An external
    /// producer that hands us `"tokenizer_plugin_library": ""` must not
    /// have that value rewritten to the reader's CWD by the
    /// `deserialize_plugin_path` hook.
    #[test]
    fn test_plugin_path_deserialize_keeps_empty_string_empty() {
        let mut json = serde_json::to_value(InvertedIndexParams::default()).unwrap();
        let obj = json.as_object_mut().unwrap();
        obj.insert(
            "tokenizer_plugin_library".to_string(),
            serde_json::Value::from(""),
        );
        let params: InvertedIndexParams = serde_json::from_value(json).unwrap();
        assert_eq!(
            params.tokenizer_plugin_library.as_deref(),
            Some(""),
            "deserializer must not absolutize an empty library path \
             into the reader's CWD"
        );
    }
}
