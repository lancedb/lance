// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{env, path::PathBuf};

#[cfg(feature = "tokenizer-jieba")]
mod jieba;

pub mod lance_tokenizer;
#[cfg(feature = "tokenizer-lindera")]
mod lindera;

#[cfg(feature = "tokenizer-jieba")]
use jieba::JiebaTokenizerBuilder;

#[cfg(feature = "tokenizer-lindera")]
use lindera::LinderaTokenizerBuilder;

use crate::pbold;
use crate::scalar::inverted::tokenizer::lance_tokenizer::{
    JsonTokenizer, LanceTokenizer, TextTokenizer,
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
    pub(crate) language: tantivy::tokenizer::Language,

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
        Self::new("simple".to_owned(), tantivy::tokenizer::Language::English)
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
    pub fn new(base_tokenizer: String, language: tantivy::tokenizer::Language) -> Self {
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

    pub fn build(&self) -> Result<Box<dyn LanceTokenizer>> {
        let mut builder = self.build_base_tokenizer()?;
        if let Some(max_token_length) = self.max_token_length {
            builder = builder.filter_dynamic(tantivy::tokenizer::RemoveLongFilter::limit(
                max_token_length,
            ));
        }
        if self.lower_case {
            builder = builder.filter_dynamic(tantivy::tokenizer::LowerCaser);
        }
        if self.stem {
            builder = builder.filter_dynamic(tantivy::tokenizer::Stemmer::new(self.language));
        }
        if self.remove_stop_words {
            let stop_word_filter = match &self.custom_stop_words {
                Some(words) => tantivy::tokenizer::StopWordFilter::remove(words.iter().cloned()),
                None => {
                    tantivy::tokenizer::StopWordFilter::new(self.language).ok_or_else(|| {
                        Error::invalid_input(format!(
                            "removing stop words for language {:?} is not supported yet",
                            self.language
                        ))
                    })?
                }
            };
            builder = builder.filter_dynamic(stop_word_filter);
        }
        if self.ascii_folding {
            builder = builder.filter_dynamic(tantivy::tokenizer::AsciiFoldingFilter);
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

    fn build_base_tokenizer(&self) -> Result<tantivy::tokenizer::TextAnalyzerBuilder> {
        match self.base_tokenizer.as_str() {
            "simple" => Ok(tantivy::tokenizer::TextAnalyzer::builder(
                tantivy::tokenizer::SimpleTokenizer::default(),
            )
            .dynamic()),
            "whitespace" => Ok(tantivy::tokenizer::TextAnalyzer::builder(
                tantivy::tokenizer::WhitespaceTokenizer::default(),
            )
            .dynamic()),
            "raw" => Ok(tantivy::tokenizer::TextAnalyzer::builder(
                tantivy::tokenizer::RawTokenizer::default(),
            )
            .dynamic()),
            "ngram" => Ok(tantivy::tokenizer::TextAnalyzer::builder(
                tantivy::tokenizer::NgramTokenizer::new(
                    self.min_ngram_length as usize,
                    self.max_ngram_length as usize,
                    self.prefix_only,
                )
                .map_err(|e| Error::invalid_input(e.to_string()))?,
            )
            .dynamic()),
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
    use super::InvertedIndexParams;

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
}
