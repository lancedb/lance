// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::path::PathBuf;

use lance_core::{Error, Result, LANCE_HOME};
use serde::{Deserialize, Serialize};
use snafu::{location, Location};

/// Tokenizer configs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// base tokenizer:
    /// - `simple`: splits tokens on whitespace and punctuation
    /// - `whitespace`: splits tokens on whitespace
    /// - `raw`: no tokenization
    /// - `lindera-ipadic`: Japanese tokenizer
    /// - `lindera-ko-dic`: Korea tokenizer
    ///
    /// `simple` is recommended for most cases and the default value
    base_tokenizer: String,

    /// language for stemming and stop words
    /// this is only used when `stem` or `remove_stop_words` is true
    language: tantivy::tokenizer::Language,

    /// maximum token length
    /// - `None`: no limit
    /// - `Some(n)`: remove tokens longer than `n`
    max_token_length: Option<usize>,

    /// whether lower case tokens
    lower_case: bool,

    /// whether apply stemming
    stem: bool,

    /// whether remove stop words
    remove_stop_words: bool,

    /// ascii folding
    ascii_folding: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self::new("simple".to_owned(), tantivy::tokenizer::Language::English)
    }
}

impl TokenizerConfig {
    pub fn new(base_tokenizer: String, language: tantivy::tokenizer::Language) -> Self {
        Self {
            base_tokenizer,
            language,
            max_token_length: Some(40),
            lower_case: true,
            stem: false,
            remove_stop_words: false,
            ascii_folding: false,
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

    pub fn build(&self) -> Result<tantivy::tokenizer::TextAnalyzer> {
        let mut builder = build_base_tokenizer_builder(&self.base_tokenizer)?;
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
}

fn build_base_tokenizer_builder(name: &str) -> Result<tantivy::tokenizer::TextAnalyzerBuilder> {
    match name {
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
        #[cfg(feature = "tokenizer-lindera")]
        s if s.starts_with("lindera/") => {
            return build_lindera_tokenizer_builder(s);
        }
        _ => Err(Error::invalid_input(
            format!("unknown base tokenizer {}", name),
            location!(),
        )),
    }
}

lazy_static::lazy_static! {
    pub static ref LANCE_TOKENIZER_HOME: Option<PathBuf> = LANCE_HOME.as_ref().map(|p| p.join("tokenizers"));
}

#[cfg(feature = "tokenizer-lindera")]
fn build_lindera_tokenizer_builder(dic: &str) -> Result<tantivy::tokenizer::TextAnalyzerBuilder> {
    use std::{fs::File, io::BufReader};

    use lindera::{
        dictionary::{
            load_dictionary_from_path, load_user_dictionary_from_config, UserDictionaryConfig,
        },
        mode::Mode,
        segmenter::Segmenter,
    };
    use lindera_tantivy::tokenizer::LinderaTokenizer;
    use serde_json::from_reader;

    match LANCE_TOKENIZER_HOME.as_ref() {
        Some(p) => {
            let dic_dir = p.join(dic);
            let main_dir = dic_dir.join("main");
            let user_config_path = dic_dir.join("user_config.json");
            let user_dictionary = if user_config_path.exists() {
                let file = File::open(user_config_path)?;
                let reader = BufReader::new(file);
                let user_dictionary_config: UserDictionaryConfig = from_reader(reader)?;
                Some(
                    load_user_dictionary_from_config(&user_dictionary_config).map_err(|e| {
                        Error::io(
                            format!("load lindera tokenizer user dictionary err: {e}"),
                            location!(),
                        )
                    })?,
                )
            } else {
                None
            };
            let mode = Mode::Normal;
            let dictionary = load_dictionary_from_path(main_dir.as_path()).map_err(|e| {
                Error::io(
                    format!("load lindera tokenizer main dictionary err: {e}"),
                    location!(),
                )
            })?;
            let segmenter = Segmenter::new(mode, dictionary, user_dictionary);
            let tokenizer = LinderaTokenizer::from_segmenter(segmenter);
            Ok(tantivy::tokenizer::TextAnalyzer::builder(tokenizer).dynamic())
        }
        None => Err(Error::invalid_input(
            format!(
                "{} is undefined",
                String::from(lance_core::LANCE_HOME_ENV_KEY)
            ),
            location!(),
        )),
    }
}
