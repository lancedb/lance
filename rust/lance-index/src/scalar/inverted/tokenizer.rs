// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{env, path::PathBuf};

use lance_core::{Error, Result};
use serde::{Deserialize, Serialize};
use snafu::location;

#[cfg(feature = "tokenizer-jieba")]
mod jieba;

#[cfg(feature = "tokenizer-lindera")]
mod lindera;

#[cfg(feature = "tokenizer-jieba")]
use jieba::JiebaTokenizerBuilder;

#[cfg(feature = "tokenizer-lindera")]
use lindera::LinderaTokenizerBuilder;

use crate::pb;

/// Tokenizer configs
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InvertedIndexParams {
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
}

impl TryFrom<&InvertedIndexParams> for pb::InvertedIndexDetails {
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
            base_tokenizer,
            language,
            with_position: false,
            max_token_length: Some(40),
            lower_case: true,
            stem: true,
            remove_stop_words: true,
            ascii_folding: true,
            min_ngram_length: default_min_ngram_length(),
            max_ngram_length: default_max_ngram_length(),
            prefix_only: false,
        }
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

    pub fn build(&self) -> Result<tantivy::tokenizer::TextAnalyzer> {
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
            let stop_word_filter = tantivy::tokenizer::StopWordFilter::new(self.language)
                .ok_or_else(|| {
                    Error::invalid_input(
                        format!(
                            "removing stop words for language {:?} is not supported yet",
                            self.language
                        ),
                        location!(),
                    )
                })?;
            builder = builder.filter_dynamic(stop_word_filter);
        }
        if self.ascii_folding {
            builder = builder.filter_dynamic(tantivy::tokenizer::AsciiFoldingFilter);
        }
        Ok(builder.build())
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
                .map_err(|e| Error::invalid_input(e.to_string(), location!()))?,
            )
            .dynamic()),
            #[cfg(feature = "tokenizer-lindera")]
            s if s.starts_with("lindera/") => {
                let Some(home) = language_model_home() else {
                    return Err(Error::invalid_input(
                        format!("unknown base tokenizer {}", self.base_tokenizer),
                        location!(),
                    ));
                };
                lindera::LinderaBuilder::load(&home.join(s))?.build()
            }
            #[cfg(feature = "tokenizer-jieba")]
            s if s.starts_with("jieba/") || s == "jieba" => {
                let s = if s == "jieba" { "jieba/default" } else { s };
                let Some(home) = language_model_home() else {
                    return Err(Error::invalid_input(
                        format!("unknown base tokenizer {}", self.base_tokenizer),
                        location!(),
                    ));
                };
                jieba::JiebaBuilder::load(&home.join(s))?.build()
            }
            _ => Err(Error::invalid_input(
                format!("unknown base tokenizer {}", self.base_tokenizer),
                location!(),
            )),
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
