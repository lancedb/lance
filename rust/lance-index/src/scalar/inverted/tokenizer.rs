// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::{Error, Result};
use serde::{Deserialize, Serialize};
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

    /// Absolute path to a tokenizer plugin shared library implementing the
    /// C ABI in `include/lance_tokenizer_plugin.h`.
    ///
    /// When set, `base_tokenizer` must be `"plugin"`.
    #[serde(default)]
    pub(crate) tokenizer_plugin_library: Option<String>,

    /// Plugin-defined configuration string (e.g. JSON, YAML) passed verbatim to the plugin.
    #[serde(default)]
    pub(crate) tokenizer_plugin_config: Option<String>,
}

impl TryFrom<&InvertedIndexParams> for pbold::InvertedIndexDetails {
    type Error = Error;

    fn try_from(params: &InvertedIndexParams) -> Result<Self> {
        params.validate_plugin_consistency()?;
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

    /// Configure a tokenizer plugin.
    /// `library_path` must be an absolute path to the plugin shared library (`.so`/`.dylib`/`.dll`).
    ///
    /// Standard filters (`max_token_length`, `lower_case`, `stem`, `remove_stop_words`, `ascii_folding`)
    /// are disabled by default for plugin tokenizers.
    /// Re-enable any Lance-side filter that should run after the plugin returns tokens.
    ///
    /// SECURITY: opening an index whose manifest carries a plugin path will
    /// `dlopen` the referenced library and execute its code in the host
    /// process. Only enable this for indexes from sources you trust as much as the host.
    pub fn plugin(mut self, library_path: String, config: String) -> Self {
        self.base_tokenizer = "plugin".to_string();
        self.tokenizer_plugin_library = Some(library_path);
        self.tokenizer_plugin_config = Some(config);
        self.max_token_length = None;
        self.lower_case = false;
        self.stem = false;
        self.remove_stop_words = false;
        self.ascii_folding = false;
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

    /// Single validation point for plugin fields, called from both `build()`
    /// and the proto-conversion path so callers see a stable failure mode
    /// regardless of how the params were constructed:
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
        if let Some(path) = self.tokenizer_plugin_library.as_deref() {
            if path.is_empty() {
                return Err(Error::invalid_input(
                    "tokenizer_plugin_library is set to an empty string; \
                     provide an absolute path to the plugin shared library",
                ));
            }
            if !std::path::Path::new(path).is_absolute() {
                return Err(Error::invalid_input(format!(
                    "tokenizer_plugin_library must be an absolute path, got {:?}. \
                     Relative paths are rejected because they would resolve \
                     against whatever process reopens the index — pass the \
                     fully-qualified path to the plugin shared library.",
                    path,
                )));
            }
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

            let config = self.tokenizer_plugin_config.as_ref().ok_or_else(|| {
                Error::invalid_input(
                    "base_tokenizer is 'plugin' but tokenizer_plugin_config is not set.",
                )
            })?;

            let tokenizer = PluginTokenizer::new(plugin_path, config)?;
            Ok(TextAnalyzer::builder(tokenizer).dynamic())
        }

        #[cfg(not(feature = "tokenizer-plugin"))]
        Err(Error::invalid_input(
            "tokenizer-plugin feature is not enabled, cannot use plugin tokenizers",
        ))
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
    use super::{
        InvertedIndexParams, Language, default_max_ngram_length, default_min_ngram_length,
    };
    use crate::pbold;

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

    /// Plugin fields without `base_tokenizer == "plugin"` must fail loud:
    /// otherwise indexing would use `simple` while metadata claimed a plugin.
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

    #[test]
    fn test_build_default_params_still_succeeds() {
        let params = InvertedIndexParams::default();
        params
            .build()
            .expect("default params must still build a simple tokenizer");
    }

    /// Reject empty paths with a self-explanatory error rather than letting
    /// `libloading::Library::new("")` surface an opaque platform error.
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

    /// Relative paths would resolve against the reader's CWD — silently
    /// loading a different file. Reject at validation with a clear error.
    #[cfg(feature = "tokenizer-plugin")]
    #[test]
    fn test_build_rejects_relative_plugin_library_path() {
        let params = InvertedIndexParams {
            base_tokenizer: "plugin".to_string(),
            tokenizer_plugin_library: Some("relative/dir/lib.so".to_string()),
            tokenizer_plugin_config: Some("{}".to_string()),
            ..InvertedIndexParams::default()
        };

        match params.build() {
            Ok(_) => panic!("build must reject a relative plugin library path"),
            Err(err) => {
                let msg = err.to_string();
                assert!(
                    msg.contains("absolute"),
                    "error must call out the absolute-path requirement, got: {}",
                    msg
                );
            }
        }
    }

    /// Pin that empty paths stay empty through `plugin()` and serde — guards
    /// against a future "prepend CWD if not absolute" rewrite that would
    /// silently defeat the empty-path validation.
    #[test]
    fn test_plugin_builder_keeps_empty_path_empty() {
        let params = InvertedIndexParams::default().plugin(String::new(), "{}".to_string());
        assert_eq!(params.tokenizer_plugin_library.as_deref(), Some(""));
    }

    #[test]
    fn test_plugin_builder_disables_standard_filters_by_default() {
        let params =
            InvertedIndexParams::default().plugin("/tmp/lib.so".to_string(), "{}".to_string());

        assert_eq!(params.max_token_length, None);
        assert!(!params.lower_case);
        assert!(!params.stem);
        assert!(!params.remove_stop_words);
        assert!(!params.ascii_folding);
    }

    #[test]
    fn test_plugin_builder_allows_reenabling_standard_filters() {
        let params = InvertedIndexParams::default()
            .plugin("/tmp/lib.so".to_string(), "{}".to_string())
            .max_token_length(Some(40))
            .lower_case(true)
            .stem(true)
            .remove_stop_words(true)
            .ascii_folding(true);

        assert_eq!(params.max_token_length, Some(40));
        assert!(params.lower_case);
        assert!(params.stem);
        assert!(params.remove_stop_words);
        assert!(params.ascii_folding);
    }

    #[test]
    fn test_plugin_path_deserialize_keeps_empty_string_empty() {
        let mut json = serde_json::to_value(InvertedIndexParams::default()).unwrap();
        let obj = json.as_object_mut().unwrap();
        obj.insert(
            "tokenizer_plugin_library".to_string(),
            serde_json::Value::from(""),
        );
        let params: InvertedIndexParams = serde_json::from_value(json).unwrap();
        assert_eq!(params.tokenizer_plugin_library.as_deref(), Some(""));
    }

    /// Catch a stray relative path at persistence even if it slipped past
    /// the build path (hand-built params serialized without going through
    /// `plugin()` + `build()`).
    #[test]
    fn test_proto_serialize_rejects_relative_plugin_path() {
        let params = InvertedIndexParams {
            base_tokenizer: "plugin".to_string(),
            tokenizer_plugin_library: Some("relative/dir/lib.so".to_string()),
            tokenizer_plugin_config: Some("{}".to_string()),
            ..InvertedIndexParams::default()
        };
        let result = pbold::InvertedIndexDetails::try_from(&params);
        let err = result.expect_err("proto conversion must reject a relative plugin path");
        let msg = err.to_string();
        assert!(
            msg.contains("absolute"),
            "error must mention the absolute-path requirement, got: {}",
            msg
        );
    }

    /// Counterpart: empty proto path stays empty after `TryFrom` so the
    /// downstream empty-path reject still fires.
    #[test]
    fn test_proto_deserialize_keeps_empty_plugin_path_empty() {
        let details = pbold::InvertedIndexDetails {
            base_tokenizer: Some("plugin".to_string()),
            language: serde_json::to_string(&Language::English).unwrap(),
            with_position: false,
            max_token_length: None,
            lower_case: true,
            stem: true,
            remove_stop_words: true,
            ascii_folding: true,
            min_ngram_length: default_min_ngram_length(),
            max_ngram_length: default_max_ngram_length(),
            prefix_only: false,
            tokenizer_plugin_library: Some(String::new()),
            tokenizer_plugin_config: Some("{}".to_string()),
        };

        let params = InvertedIndexParams::try_from(&details).expect("proto roundtrip");
        assert_eq!(params.tokenizer_plugin_library.as_deref(), Some(""));
    }
}
