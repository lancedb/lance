// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{env, path::PathBuf};

use lance_core::{Error, Result};
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
        #[cfg(feature = "tokenizer-jieba")]
        s if s.starts_with("jieba/") || s == "jieba" => {
            return build_jieba_tokenizer_builder(s);
        }
        _ => Err(Error::invalid_input(
            format!("unknown base tokenizer {}", name),
            location!(),
        )),
    }
}

pub const LANCE_LANGUAGE_MODEL_HOME_ENV_KEY: &str = "LANCE_LANGUAGE_MODEL_HOME";

pub const LANCE_LANGUAGE_MODEL_DEFAULT_DIRECTORY: &str = "lance/language_models";

lazy_static::lazy_static! {
    /// default directory that stores lance tokenizer related files, e.g. tokenizer model.
    pub static ref LANCE_LANGUAGE_MODEL_HOME: Option<PathBuf> = match env::var(LANCE_LANGUAGE_MODEL_HOME_ENV_KEY) {
        Ok(p) => Some(PathBuf::from(p)),
        Err(_) => dirs::data_local_dir().map(|p| p.join(LANCE_LANGUAGE_MODEL_DEFAULT_DIRECTORY))
    };
}

#[cfg(feature = "tokenizer-lindera")]
#[derive(Serialize, Deserialize)]
struct LinderaConfig{
    main: String,
    user: Option<String>,
    user_kind: Option<String>
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
    use serde_json::Value;

    match LANCE_LANGUAGE_MODEL_HOME.as_ref() {
        Some(p) => {
            let dic_dir = p.join(dic);
            let config_path = dic_dir.join("config.json");
            let config: LinderaConfig = if config_path.exists() {
                let file = File::open(config_path)?;
                let reader = BufReader::new(file);
                serde_json::from_reader(reader)?
            } else {
                let Some(dic_dir) = dic_dir.to_str() else {
                    return Err(Error::invalid_input("dic dir is invalid",
                        location!(),
                    ))
                };
                LinderaConfig{main: String::from(dic_dir), user: None, user_kind: None}
            };
            let main_path = dic_dir.join(config.main);
            let dictionary = load_dictionary_from_path(main_path.as_path()).map_err(|e| {
                Error::io(
                    format!("load lindera tokenizer main dictionary err: {e}"),
                    location!(),
                )
            })?;
            let user_dictionary = match config.user {
                Some(user) =>  {
                    let mut conf = serde_json::Map::<String, Value>::new();
                    let user_path = dic_dir.join(user);
                    match user_path.to_str() {
                        Some(p) => {
                            conf.insert(String::from("path"), Value::String(String::from(p)));
                            Ok(())
                        },
                        None => {
                            let p = user_path.display();
                            Err(Error::io(
                                format!("invalid lindera tokenizer user dictionary path: {p}"),
                                location!(),
                            ))
                        }
                    }?;
                    if let Some(kind) = config.user_kind {
                        conf.insert(String::from("kind"), Value::String(kind));
                    }
                    let user_dictionary_config: UserDictionaryConfig = Value::Object(conf);
                    let user_dictionary = load_user_dictionary_from_config(&user_dictionary_config).map_err(|e| {
                        Error::io(
                            format!("load lindera tokenizer user dictionary err: {e}"),
                            location!(),
                        )
                    })?;
                    Some(user_dictionary)
                },
                None => None

            };
            let mode = Mode::Normal;
            let segmenter = Segmenter::new(mode, dictionary, user_dictionary);
            let tokenizer = LinderaTokenizer::from_segmenter(segmenter);
            Ok(tantivy::tokenizer::TextAnalyzer::builder(tokenizer).dynamic())
        }
        None => Err(Error::invalid_input(
            format!(
                "{} is undefined",
                String::from(LANCE_LANGUAGE_MODEL_HOME_ENV_KEY)
            ),
            location!(),
        )),
    }
}



#[cfg(feature = "tokenizer-jieba")]
fn build_jieba_tokenizer_builder(dic: &str) -> Result<tantivy::tokenizer::TextAnalyzerBuilder> {
    match LANCE_LANGUAGE_MODEL_HOME.as_ref() {
        Some(p) => {
            let dic = if dic == "jieba" {
                "jieba/default"
            } else {
                dic
            };
            let dic_file = p.join(dic).join("dict.txt");
            let file = std::fs::File::open(dic_file)?;
            let mut f = std::io::BufReader::new(file);
            let jieba = jieba_rs::Jieba::with_dict(&mut f).map_err(|e| {
                Error::io(
                    format!("load jieba tokenizer dictionary err: {e}"),
                    location!(),
                )
            })?;
            let tokenizer = JiebaTokenizer{jieba: jieba};
            Ok(tantivy::tokenizer::TextAnalyzer::builder(tokenizer).dynamic())
        },
        None => Err(Error::invalid_input(
            format!(
                "{} is undefined",
                String::from(LANCE_LANGUAGE_MODEL_HOME_ENV_KEY)
            ),
            location!(),
        )),
    }

}

#[cfg(feature = "tokenizer-jieba")]
#[derive(Clone)]
struct JiebaTokenizer{
    jieba: jieba_rs::Jieba
}

#[cfg(feature = "tokenizer-jieba")]
struct JiebaTokenStream {
    tokens: Vec<tantivy::tokenizer::Token>,
    index: usize,
}

#[cfg(feature = "tokenizer-jieba")]
impl tantivy::tokenizer::TokenStream for JiebaTokenStream {
    fn advance(&mut self) -> bool {
        if self.index < self.tokens.len() {
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn token(&self) -> &tantivy::tokenizer::Token {
        &self.tokens[self.index - 1]
    }

    fn token_mut(&mut self) -> &mut tantivy::tokenizer::Token {
        &mut self.tokens[self.index - 1]
    }
}


#[cfg(feature = "tokenizer-jieba")]
impl tantivy::tokenizer::Tokenizer for JiebaTokenizer {
    type TokenStream<'a> = JiebaTokenStream;

    fn token_stream(&mut self, text: &str) -> JiebaTokenStream {
        let mut indices = text.char_indices().collect::<Vec<_>>();
        indices.push((text.len(), '\0'));
        let orig_tokens = self.jieba.tokenize(text, jieba_rs::TokenizeMode::Search, true);
        let mut tokens = Vec::new();
        for token in orig_tokens {
            tokens.push(tantivy::tokenizer::Token {
                offset_from: indices[token.start].0,
                offset_to: indices[token.end].0,
                position: token.start,
                text: String::from(&text[(indices[token.start].0)..(indices[token.end].0)]),
                position_length: token.end - token.start,
            });
        }
        JiebaTokenStream { tokens, index: 0 }
    }
}
