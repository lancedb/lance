// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::path::{Path, PathBuf};

use super::TokenizerBuilder;
use lance_core::{Error, Result};
use serde::{Deserialize, Serialize};
use snafu::location;

#[derive(Serialize, Deserialize, Default)]
pub struct JiebaConfig {
    main: Option<String>,
    users: Option<Vec<String>>,
}

pub struct JiebaBuilder {
    root: PathBuf,
    config: JiebaConfig,
}

impl JiebaBuilder {
    fn main_dict_path(&self) -> PathBuf {
        if let Some(p) = &self.config.main {
            return self.root.join(p);
        }
        self.root.join("dict.txt")
    }

    fn user_dict_paths(&self) -> Vec<PathBuf> {
        let Some(users) = &self.config.users else {
            return vec![];
        };
        users.iter().map(|p| self.root.join(p)).collect()
    }
}

impl TokenizerBuilder for JiebaBuilder {
    type Config = JiebaConfig;

    fn new(config: Self::Config, root: &Path) -> Result<Self> {
        Ok(Self {
            config,
            root: root.to_path_buf(),
        })
    }

    fn build(&self) -> Result<tantivy::tokenizer::TextAnalyzerBuilder> {
        let main_dict_path = &self.main_dict_path();
        let file = std::fs::File::open(main_dict_path)?;
        let mut f = std::io::BufReader::new(file);
        let mut jieba = jieba_rs::Jieba::with_dict(&mut f).map_err(|e| {
            Error::io(
                format!(
                    "load jieba tokenizer dictionary {}, error: {}",
                    main_dict_path.display(),
                    e
                ),
                location!(),
            )
        })?;
        for user_dict_path in &self.user_dict_paths() {
            let file = std::fs::File::open(user_dict_path)?;
            let mut f = std::io::BufReader::new(file);
            jieba.load_dict(&mut f).map_err(|e| {
                Error::io(
                    format!(
                        "load jieba tokenizer user dictionary {},  error: {}",
                        user_dict_path.display(),
                        e
                    ),
                    location!(),
                )
            })?
        }
        let tokenizer = JiebaTokenizer { jieba };
        Ok(tantivy::tokenizer::TextAnalyzer::builder(tokenizer).dynamic())
    }
}

#[derive(Clone)]
struct JiebaTokenizer {
    jieba: jieba_rs::Jieba,
}

struct JiebaTokenStream {
    tokens: Vec<tantivy::tokenizer::Token>,
    index: usize,
}

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
        let orig_tokens = self
            .jieba
            .tokenize(text, jieba_rs::TokenizeMode::Search, true);
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
