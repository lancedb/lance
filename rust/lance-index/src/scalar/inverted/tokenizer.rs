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

#[cfg(feature = "tokenizer-jieba")]
use jieba::JiebaTokenizerBuilder;

#[cfg(feature = "tokenizer-lindera")]
use lindera::LinderaTokenizerBuilder;

use crate::pbold;
use crate::scalar::inverted::tokenizer::document_tokenizer::{
    JsonTokenizer, LanceTokenizer, TextTokenizer,
};
use crate::scalar::inverted::{
    InvertedListFormatVersion, default_fts_format_version, resolve_fts_format_version,
};
pub use lance_tokenizer::Language;
use lance_tokenizer::{
    AsciiFoldingFilter, CodeLexTokenizer, IcuTokenizer, LowerCaser, NgramTokenizer, RawTokenizer,
    RemoveLongFilter, SimpleTokenizer, Stemmer, StopWordFilter, TextAnalyzer, TextAnalyzerBuilder,
    WhitespaceTokenizer, WordDelimiterFilter,
};

/// Tokenizer configs
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct InvertedIndexParams {
    /// High-level analyzer profile.
    ///
    /// This is the user-facing semantic preset. `text` keeps the existing
    /// natural-language defaults. `code` selects code-search defaults such as
    /// code lexical tokenization, identifier splitting, preserving whole
    /// identifiers, and disabling stemming / stop-word removal.
    pub(crate) analyzer: String,

    /// Document-level tokenizer.
    ///
    /// This decides how Lance extracts searchable text from the stored value:
    /// - `text`: plain string documents
    /// - `json`: JSON string documents
    /// - `None`: infer from the Arrow field type during index build
    ///
    /// The extracted text is then passed to `base_tokenizer`.
    pub(crate) lance_tokenizer: Option<String>,

    /// Lexical tokenizer used inside the selected analyzer profile.
    ///
    /// This is a lower-level implementation component, not the product-level
    /// search mode. For code search, prefer `analyzer = "code"`;
    /// `base_tokenizer = "code"` is accepted as a compatibility alias and
    /// resolves to the code analyzer profile.
    /// - `simple`: splits tokens on whitespace and punctuation
    /// - `whitespace`: splits tokens on whitespace
    /// - `raw`: no tokenization
    /// - `code`: code-aware lexical tokenization
    /// - `icu`: ICU dictionary-based word segmentation
    /// - `icu/split`: ICU segmentation with simple-style delimiter splitting
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
    pub(crate) with_position: bool,

    /// maximum token length
    /// - `None`: no limit
    /// - `Some(n)`: remove tokens longer than `n`
    pub(crate) max_token_length: Option<usize>,

    /// whether lower case tokens
    pub(crate) lower_case: bool,

    /// whether apply stemming
    pub(crate) stem: bool,

    /// whether remove stop words
    pub(crate) remove_stop_words: bool,

    /// use customized stop words.
    /// - `None`: use built-in stop words based on language
    /// - `Some(words)`: use customized stop words
    pub(crate) custom_stop_words: Option<Vec<String>>,

    /// ascii folding
    pub(crate) ascii_folding: bool,

    /// min ngram length
    pub(crate) min_ngram_length: u32,

    /// max ngram length
    pub(crate) max_ngram_length: u32,

    /// whether prefix only
    pub(crate) prefix_only: bool,

    /// Split code identifiers into subwords.
    pub(crate) split_identifiers: bool,

    /// Split identifier subwords at letter/number boundaries.
    pub(crate) split_on_numerics: bool,

    /// Index a complete identifier in addition to subwords.
    pub(crate) preserve_original: bool,

    /// Index code operators such as `::`, `->`, and `!=`.
    pub(crate) index_operators: bool,

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

    /// On-disk FTS format version to write when creating a new index.
    ///
    /// This is a build-time only parameter and is not persisted with the index.
    /// If unset, Lance writes the current default FTS format.
    #[serde(
        rename = "format_version",
        skip_serializing,
        default,
        deserialize_with = "deserialize_format_version"
    )]
    pub(crate) format_version: Option<InvertedListFormatVersion>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawInvertedIndexParams {
    analyzer: Option<String>,
    lance_tokenizer: Option<String>,
    base_tokenizer: Option<String>,
    language: Option<Language>,
    with_position: Option<bool>,
    max_token_length: Option<Option<usize>>,
    lower_case: Option<bool>,
    stem: Option<bool>,
    remove_stop_words: Option<bool>,
    custom_stop_words: Option<Vec<String>>,
    ascii_folding: Option<bool>,
    min_ngram_length: Option<u32>,
    max_ngram_length: Option<u32>,
    prefix_only: Option<bool>,
    split_identifiers: Option<bool>,
    split_on_numerics: Option<bool>,
    preserve_original: Option<bool>,
    index_operators: Option<bool>,
    #[serde(rename = "memory_limit", alias = "worker_memory_limit_mb")]
    memory_limit_mb: Option<u64>,
    #[serde(rename = "num_workers")]
    num_workers: Option<usize>,
    #[serde(
        rename = "format_version",
        default,
        deserialize_with = "deserialize_format_version"
    )]
    format_version: Option<InvertedListFormatVersion>,
}

impl<'de> Deserialize<'de> for InvertedIndexParams {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        RawInvertedIndexParams::deserialize(deserializer)?
            .resolve()
            .map_err(serde::de::Error::custom)
    }
}

impl RawInvertedIndexParams {
    fn resolve(self) -> Result<InvertedIndexParams> {
        let analyzer = match (
            self.analyzer.as_deref().map(normalize_analyzer),
            self.base_tokenizer.as_deref(),
        ) {
            (Some(Ok(analyzer)), _) => analyzer,
            (Some(Err(err)), _) => return Err(err),
            (None, Some("code")) => "code",
            (None, _) => "text",
        };

        if analyzer == "text" && self.base_tokenizer.as_deref() == Some("code") {
            return Err(Error::invalid_input(
                "base_tokenizer='code' requires analyzer='code'".to_string(),
            ));
        }
        if analyzer == "code"
            && let Some(base_tokenizer) = self.base_tokenizer.as_deref()
            && base_tokenizer != "code"
        {
            return Err(Error::invalid_input(format!(
                "analyzer='code' requires base_tokenizer='code', got '{}'",
                base_tokenizer
            )));
        }
        if analyzer == "text"
            && matches!(
                (
                    self.split_identifiers,
                    self.split_on_numerics,
                    self.preserve_original,
                    self.index_operators,
                ),
                (Some(true), _, _, _)
                    | (_, Some(true), _, _)
                    | (_, _, Some(true), _)
                    | (_, _, _, Some(true))
            )
        {
            return Err(Error::invalid_input(
                "code analyzer flags require analyzer='code'".to_string(),
            ));
        }

        let mut params = match analyzer {
            "code" => InvertedIndexParams::code(),
            "text" => InvertedIndexParams::default(),
            _ => unreachable!("analyzer is normalized above"),
        };

        params.analyzer = analyzer.to_string();
        if let Some(lance_tokenizer) = self.lance_tokenizer {
            params.lance_tokenizer = Some(lance_tokenizer);
        }
        if let Some(base_tokenizer) = self.base_tokenizer {
            params.base_tokenizer = base_tokenizer;
        }
        if let Some(language) = self.language {
            params.language = language;
        }
        if let Some(with_position) = self.with_position {
            params.with_position = with_position;
        }
        if let Some(max_token_length) = self.max_token_length {
            params.max_token_length = max_token_length;
        }
        if let Some(lower_case) = self.lower_case {
            params.lower_case = lower_case;
        }
        if let Some(stem) = self.stem {
            params.stem = stem;
        }
        if let Some(remove_stop_words) = self.remove_stop_words {
            params.remove_stop_words = remove_stop_words;
        }
        if let Some(custom_stop_words) = self.custom_stop_words {
            params.custom_stop_words = Some(custom_stop_words);
        }
        if let Some(ascii_folding) = self.ascii_folding {
            params.ascii_folding = ascii_folding;
        }
        if let Some(min_ngram_length) = self.min_ngram_length {
            params.min_ngram_length = min_ngram_length;
        }
        if let Some(max_ngram_length) = self.max_ngram_length {
            params.max_ngram_length = max_ngram_length;
        }
        if let Some(prefix_only) = self.prefix_only {
            params.prefix_only = prefix_only;
        }
        if let Some(split_identifiers) = self.split_identifiers {
            params.split_identifiers = split_identifiers;
        }
        if let Some(split_on_numerics) = self.split_on_numerics {
            params.split_on_numerics = split_on_numerics;
        }
        if let Some(preserve_original) = self.preserve_original {
            params.preserve_original = preserve_original;
        }
        if let Some(index_operators) = self.index_operators {
            params.index_operators = index_operators;
        }
        params.memory_limit_mb = self.memory_limit_mb;
        params.num_workers = self.num_workers;
        params.format_version = self.format_version;
        params.validate()?;
        Ok(params)
    }
}

impl TryFrom<&InvertedIndexParams> for pbold::InvertedIndexDetails {
    type Error = Error;

    fn try_from(params: &InvertedIndexParams) -> Result<Self> {
        Ok(Self {
            analyzer: Some(params.analyzer.clone()),
            lance_tokenizer: params.lance_tokenizer.clone(),
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
            split_identifiers: params.split_identifiers,
            split_on_numerics: params.split_on_numerics,
            preserve_original: params.preserve_original,
            index_operators: params.index_operators,
        })
    }
}

impl TryFrom<&pbold::InvertedIndexDetails> for InvertedIndexParams {
    type Error = Error;

    fn try_from(details: &pbold::InvertedIndexDetails) -> Result<Self> {
        if details.base_tokenizer.is_none() && details.language.is_empty() {
            return Ok(Self::default());
        }

        let analyzer = details
            .analyzer
            .as_deref()
            .map(normalize_analyzer)
            .transpose()?
            .unwrap_or_else(|| {
                if details.base_tokenizer.as_deref() == Some("code") {
                    "code"
                } else {
                    "text"
                }
            });
        let mut params = match analyzer {
            "code" => Self::code(),
            "text" => Self::default(),
            _ => unreachable!("analyzer is normalized above"),
        };
        params.analyzer = analyzer.to_string();
        params.lance_tokenizer = details.lance_tokenizer.clone();
        params.base_tokenizer = details
            .base_tokenizer
            .as_ref()
            .cloned()
            .unwrap_or_else(|| params.base_tokenizer.clone());
        params.language = if details.language.is_empty() {
            params.language
        } else {
            serde_json::from_str(details.language.as_str())?
        };
        params.with_position = details.with_position;
        params.max_token_length = details.max_token_length.map(|l| l as usize);
        params.lower_case = details.lower_case;
        params.stem = details.stem;
        params.remove_stop_words = details.remove_stop_words;
        params.ascii_folding = details.ascii_folding;
        params.min_ngram_length = details.min_ngram_length;
        params.max_ngram_length = details.max_ngram_length;
        params.prefix_only = details.prefix_only;
        params.split_identifiers = details.split_identifiers;
        params.split_on_numerics = details.split_on_numerics;
        params.preserve_original = details.preserve_original;
        params.index_operators = details.index_operators;
        params.validate()?;
        Ok(params)
    }
}

fn normalize_analyzer(analyzer: &str) -> Result<&'static str> {
    match analyzer {
        "text" => Ok("text"),
        "code" => Ok("code"),
        other => Err(Error::invalid_input(format!(
            "unknown analyzer '{}', expected 'text' or 'code'",
            other
        ))),
    }
}

fn default_min_ngram_length() -> u32 {
    3
}

fn default_max_ngram_length() -> u32 {
    3
}

fn deserialize_format_version<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<InvertedListFormatVersion>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    let Some(value) = value else {
        return Ok(None);
    };
    match value {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::String(value) => resolve_fts_format_version(Some(&value))
            .map(Some)
            .map_err(serde::de::Error::custom),
        serde_json::Value::Number(value) => {
            let Some(value) = value.as_u64() else {
                return Err(serde::de::Error::custom(
                    "FTS format_version must be 1 or 2",
                ));
            };
            resolve_fts_format_version(Some(&value.to_string()))
                .map(Some)
                .map_err(serde::de::Error::custom)
        }
        other => Err(serde::de::Error::custom(format!(
            "FTS format_version must be 1 or 2, got {other}"
        ))),
    }
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
    /// - `code`: code identifier tokenization
    /// - `ngram`: N-Gram tokenizer
    /// - `icu`: ICU dictionary-based word segmentation
    /// - `icu/split`: ICU segmentation with simple-style delimiter splitting
    /// - `lindera/*`: Lindera tokenizer
    /// - `jieba/*`: Jieba tokenizer
    ///
    /// The `language` is used for stemming and removing stop words,
    /// this is not used for `lindera/*` and `jieba/*` tokenizers.
    /// Default to `English`.
    pub fn new(base_tokenizer: String, language: Language) -> Self {
        Self {
            analyzer: "text".to_string(),
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
            split_identifiers: false,
            split_on_numerics: false,
            preserve_original: false,
            index_operators: false,
            memory_limit_mb: None,
            num_workers: None,
            format_version: None,
        }
    }

    /// Create parameters for the code analyzer profile.
    pub fn code() -> Self {
        Self::new("code".to_string(), Language::English)
            .analyzer("code")
            .expect("code analyzer should be valid")
    }

    /// Set analyzer profile.
    pub fn analyzer(mut self, analyzer: &str) -> Result<Self> {
        self.analyzer = normalize_analyzer(analyzer)?.to_string();
        if self.analyzer == "code" {
            self.base_tokenizer = "code".to_string();
            self.split_identifiers = true;
            self.split_on_numerics = true;
            self.preserve_original = true;
            self.lower_case = true;
            self.ascii_folding = true;
            self.stem = false;
            self.remove_stop_words = false;
            self.index_operators = false;
        }
        Ok(self)
    }

    pub fn lance_tokenizer(mut self, lance_tokenizer: String) -> Self {
        self.lance_tokenizer = Some(lance_tokenizer);
        self
    }

    pub fn base_tokenizer(mut self, base_tokenizer: String) -> Self {
        self.base_tokenizer = base_tokenizer;
        if self.base_tokenizer == "code" {
            self = self
                .analyzer("code")
                .expect("code analyzer should be valid");
        }
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

    pub fn split_identifiers(mut self, split_identifiers: bool) -> Self {
        self.split_identifiers = split_identifiers;
        self
    }

    pub fn split_on_numerics(mut self, split_on_numerics: bool) -> Self {
        self.split_on_numerics = split_on_numerics;
        self
    }

    pub fn preserve_original(mut self, preserve_original: bool) -> Self {
        self.preserve_original = preserve_original;
        self
    }

    pub fn index_operators(mut self, index_operators: bool) -> Self {
        self.index_operators = index_operators;
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

    /// Set the on-disk FTS format version to use when creating a new index.
    ///
    /// If unset, Lance writes the current default FTS format. Existing indexes
    /// keep their own on-disk format during update and optimize operations.
    pub fn format_version(mut self, format_version: InvertedListFormatVersion) -> Self {
        self.format_version = Some(format_version);
        self
    }

    /// Resolve the requested FTS format version, falling back to Lance's default.
    pub fn resolved_format_version(&self) -> InvertedListFormatVersion {
        self.format_version
            .unwrap_or_else(default_fts_format_version)
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
        if let Some(format_version) = self.format_version {
            object.insert(
                "format_version".to_string(),
                serde_json::Value::from(format_version.index_version()),
            );
        }
        Ok(value)
    }

    pub fn build(&self) -> Result<Box<dyn LanceTokenizer>> {
        self.validate()?;
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
            builder = builder.filter_dynamic(self.stop_word_filter()?);
        }
        if self.ascii_folding {
            builder = builder.filter_dynamic(AsciiFoldingFilter);
        }
        let tokenizer = builder.build();

        match self.lance_tokenizer {
            Some(ref t) if t == "text" => Ok(Box::new(TextTokenizer::new(tokenizer))),
            Some(ref t) if t == "json" => Ok(Box::new(JsonTokenizer::new(tokenizer))),
            None => Ok(Box::new(TextTokenizer::new(tokenizer))),
            _ => Err(Error::invalid_input(format!(
                "unknown lance tokenizer {}",
                self.lance_tokenizer.as_ref().unwrap()
            ))),
        }
    }

    fn stop_word_filter(&self) -> Result<StopWordFilter> {
        match &self.custom_stop_words {
            Some(words) => Ok(StopWordFilter::remove(words.iter().cloned())),
            None if self.base_tokenizer == "icu" || self.base_tokenizer == "icu/split" => {
                Ok(StopWordFilter::all())
            }
            None => StopWordFilter::new(self.language).ok_or_else(|| {
                Error::invalid_input(format!(
                    "removing stop words for language {:?} is not supported yet",
                    self.language
                ))
            }),
        }
    }

    fn validate(&self) -> Result<()> {
        normalize_analyzer(&self.analyzer)?;
        match self.analyzer.as_str() {
            "code" if self.base_tokenizer != "code" => Err(Error::invalid_input(format!(
                "analyzer='code' requires base_tokenizer='code', got '{}'",
                self.base_tokenizer
            ))),
            "text" if self.base_tokenizer == "code" => Err(Error::invalid_input(
                "base_tokenizer='code' requires analyzer='code'".to_string(),
            )),
            "text"
                if self.split_identifiers
                    || self.split_on_numerics
                    || self.preserve_original
                    || self.index_operators =>
            {
                Err(Error::invalid_input(
                    "code analyzer flags require analyzer='code'".to_string(),
                ))
            }
            _ => Ok(()),
        }
    }

    fn build_base_tokenizer(&self) -> Result<TextAnalyzerBuilder> {
        match self.base_tokenizer.as_str() {
            "simple" => Ok(TextAnalyzer::builder(SimpleTokenizer::default()).dynamic()),
            "whitespace" => Ok(TextAnalyzer::builder(WhitespaceTokenizer::default()).dynamic()),
            "raw" => Ok(TextAnalyzer::builder(RawTokenizer::default()).dynamic()),
            "code" => {
                let mut builder =
                    TextAnalyzer::builder(CodeLexTokenizer::new(self.index_operators)).dynamic();
                if self.split_identifiers {
                    builder = builder.filter_dynamic(WordDelimiterFilter::new(
                        self.preserve_original,
                        self.split_on_numerics,
                    ));
                }
                Ok(builder)
            }
            "icu" => Ok(TextAnalyzer::builder(IcuTokenizer::default()).dynamic()),
            "icu/split" => {
                Ok(TextAnalyzer::builder(IcuTokenizer::default().with_simple_split()).dynamic())
            }
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
            _ => Err(Error::invalid_input(format!(
                "unknown base tokenizer {}",
                self.base_tokenizer
            ))),
        }
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
    use super::{InvertedIndexParams, InvertedListFormatVersion};
    use lance_tokenizer::TokenStream;
    use rstest::rstest;
    use serde_json::json;

    #[test]
    fn test_build_only_fields_are_not_serialized() {
        let params = InvertedIndexParams::default()
            .memory_limit_mb(4096)
            .num_workers(7)
            .format_version(InvertedListFormatVersion::V1);
        let json = serde_json::to_value(&params).unwrap();
        assert!(json.get("memory_limit").is_none());
        assert!(json.get("num_workers").is_none());
        assert!(json.get("format_version").is_none());
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
        obj.insert("format_version".to_string(), serde_json::Value::from("v1"));

        let params: InvertedIndexParams = serde_json::from_value(json).unwrap();
        assert_eq!(params.memory_limit_mb, Some(4096));
        assert_eq!(params.num_workers, Some(3));
        assert_eq!(params.format_version, Some(InvertedListFormatVersion::V1));
    }

    #[test]
    fn test_training_json_serializes_build_only_fields() {
        let params = InvertedIndexParams::default()
            .memory_limit_mb(4096)
            .num_workers(3)
            .format_version(InvertedListFormatVersion::V1);
        let json = params.to_training_json().unwrap();
        assert_eq!(
            json.get("memory_limit"),
            Some(&serde_json::Value::from(4096))
        );
        assert_eq!(json.get("num_workers"), Some(&serde_json::Value::from(3)));
        assert_eq!(
            json.get("format_version"),
            Some(&serde_json::Value::from(1))
        );
    }

    #[test]
    fn test_default_format_version_resolves_to_v2() {
        assert_eq!(
            InvertedIndexParams::default().resolved_format_version(),
            InvertedListFormatVersion::V2
        );
    }

    #[test]
    fn test_code_analyzer_resolves_defaults() {
        let params: InvertedIndexParams = serde_json::from_value(json!({
            "analyzer": "code"
        }))
        .unwrap();
        assert_eq!(params.analyzer, "code");
        assert_eq!(params.base_tokenizer, "code");
        assert!(params.split_identifiers);
        assert!(params.split_on_numerics);
        assert!(params.preserve_original);
        assert!(params.lower_case);
        assert!(params.ascii_folding);
        assert!(!params.stem);
        assert!(!params.remove_stop_words);
        assert!(!params.index_operators);
    }

    #[test]
    fn test_code_base_tokenizer_alias_resolves_analyzer() {
        let params: InvertedIndexParams = serde_json::from_value(json!({
            "base_tokenizer": "code"
        }))
        .unwrap();
        assert_eq!(params.analyzer, "code");
        assert_eq!(params.base_tokenizer, "code");
    }

    #[test]
    fn test_code_analyzer_rejects_conflicting_base_tokenizer() {
        let err = serde_json::from_value::<InvertedIndexParams>(json!({
            "analyzer": "code",
            "base_tokenizer": "simple"
        }))
        .unwrap_err();
        assert!(err.to_string().contains("requires base_tokenizer='code'"));
    }

    #[test]
    fn test_build_code_tokenizer() {
        let mut tokenizer = InvertedIndexParams::code().build().unwrap();
        let mut stream = tokenizer.token_stream_for_doc(
            "getUserName XMLHttpRequest parseHTML2JSON utf8_reader SCREAMING_SNAKE_CASE",
        );
        let mut tokens = Vec::new();
        stream.process(&mut |token| {
            tokens.push((token.text.clone(), token.position, token.position_length))
        });
        assert_eq!(
            tokens,
            vec![
                ("getusername".to_string(), 0, 3),
                ("get".to_string(), 0, 1),
                ("user".to_string(), 1, 1),
                ("name".to_string(), 2, 1),
                ("xmlhttprequest".to_string(), 3, 3),
                ("xml".to_string(), 3, 1),
                ("http".to_string(), 4, 1),
                ("request".to_string(), 5, 1),
                ("parsehtml2json".to_string(), 6, 4),
                ("parse".to_string(), 6, 1),
                ("html".to_string(), 7, 1),
                ("2".to_string(), 8, 1),
                ("json".to_string(), 9, 1),
                ("utf8_reader".to_string(), 10, 3),
                ("utf".to_string(), 10, 1),
                ("8".to_string(), 11, 1),
                ("reader".to_string(), 12, 1),
                ("screaming_snake_case".to_string(), 13, 3),
                ("screaming".to_string(), 13, 1),
                ("snake".to_string(), 14, 1),
                ("case".to_string(), 15, 1),
            ]
        );
    }

    #[test]
    fn test_inverted_details_round_trip_code_params() {
        let params = InvertedIndexParams::code()
            .with_position(true)
            .index_operators(true)
            .preserve_original(false);
        let details = crate::pbold::InvertedIndexDetails::try_from(&params).unwrap();
        let round_tripped = InvertedIndexParams::try_from(&details).unwrap();
        assert_eq!(round_tripped, params);
    }

    #[test]
    fn test_build_icu_tokenizer() {
        let mut tokenizer = InvertedIndexParams::default()
            .base_tokenizer("icu".to_string())
            .stem(false)
            .remove_stop_words(false)
            .build()
            .unwrap();
        let mut stream = tokenizer.token_stream_for_doc("Hello, こんにちは世界!");
        let mut tokens = Vec::new();
        stream.process(&mut |token| tokens.push(token.text.clone()));
        assert_eq!(tokens, vec!["hello", "こんにちは", "世界"]);
    }

    #[test]
    fn test_build_icu_tokenizer_with_split_on_non_alphanumeric() {
        let mut tokenizer = InvertedIndexParams::default()
            .base_tokenizer("icu/split".to_string())
            .stem(false)
            .remove_stop_words(false)
            .build()
            .unwrap();
        let mut stream = tokenizer.token_stream_for_doc("hello_world こんにちは世界 alpha.beta");
        let mut tokens = Vec::new();
        stream.process(&mut |token| tokens.push(token.text.clone()));
        assert_eq!(
            tokens,
            vec!["hello", "world", "こんにちは", "世界", "alpha", "beta"]
        );
    }

    #[test]
    fn test_remove_stop_words_respects_language_for_non_icu_tokenizer() {
        let mut tokenizer = InvertedIndexParams::default()
            .stem(false)
            .base_tokenizer("simple".to_string())
            .build()
            .unwrap();
        let mut stream = tokenizer.token_stream_for_search("the 的 lance data");
        let mut tokens = Vec::new();
        while let Some(token) = stream.next() {
            tokens.push(token.text.clone());
        }
        assert_eq!(
            tokens,
            vec!["的".to_string(), "lance".to_string(), "data".to_string()]
        );
    }

    #[test]
    fn test_custom_stop_words_replace_language_builtins() {
        let mut tokenizer = InvertedIndexParams::default()
            .stem(false)
            .custom_stop_words(Some(vec!["lance".to_string()]))
            .build()
            .unwrap();
        let mut stream = tokenizer.token_stream_for_search("the lance data");
        let mut tokens = Vec::new();
        while let Some(token) = stream.next() {
            tokens.push(token.text.clone());
        }
        assert_eq!(tokens, vec!["the".to_string(), "data".to_string()]);
    }

    #[rstest]
    #[case::icu("icu")]
    #[case::icu_split("icu/split")]
    fn test_icu_stop_words_use_all_builtin_lists(#[case] base_tokenizer: &str) {
        let mut tokenizer = InvertedIndexParams::default()
            .stem(false)
            .base_tokenizer(base_tokenizer.to_string())
            .build()
            .unwrap();
        let mut stream = tokenizer.token_stream_for_search("the 的 lance data");
        let mut tokens = Vec::new();
        while let Some(token) = stream.next() {
            tokens.push(token.text.clone());
        }
        assert_eq!(tokens, vec!["lance".to_string(), "data".to_string()]);
    }
}
