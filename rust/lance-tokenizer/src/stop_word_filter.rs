// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-License-Identifier: MIT
// Adapted from Tantivy v0.24.2 stop-word filter.
// Copyright (c) 2017-present Tantivy contributors.

#[path = "stop_word_filter/stopwords.rs"]
mod stopwords;

use std::collections::HashSet;
use std::sync::Arc;

use crate::{Language, Token, TokenFilter, TokenStream, Tokenizer};

#[derive(Clone)]
pub struct StopWordFilter {
    words: Arc<HashSet<String>>,
}

impl StopWordFilter {
    pub fn new(language: Language) -> Option<Self> {
        let words = match language {
            Language::Danish => stopwords::DANISH,
            Language::Dutch => stopwords::DUTCH,
            Language::English => &[
                "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into",
                "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then",
                "there", "these", "they", "this", "to", "was", "will", "with",
            ],
            Language::Finnish => stopwords::FINNISH,
            Language::French => stopwords::FRENCH,
            Language::German => stopwords::GERMAN,
            Language::Hungarian => stopwords::HUNGARIAN,
            Language::Italian => stopwords::ITALIAN,
            Language::Norwegian => stopwords::NORWEGIAN,
            Language::Portuguese => stopwords::PORTUGUESE,
            Language::Russian => stopwords::RUSSIAN,
            Language::Spanish => stopwords::SPANISH,
            Language::Swedish => stopwords::SWEDISH,
            _ => return None,
        };
        Some(Self::remove(words.iter().map(|word| (*word).to_owned())))
    }

    pub fn remove<W: IntoIterator<Item = String>>(words: W) -> Self {
        Self {
            words: Arc::new(words.into_iter().collect()),
        }
    }
}

impl TokenFilter for StopWordFilter {
    type Tokenizer<T: Tokenizer> = StopWordFilterWrapper<T>;

    fn transform<T: Tokenizer>(self, tokenizer: T) -> Self::Tokenizer<T> {
        StopWordFilterWrapper {
            words: self.words,
            inner: tokenizer,
        }
    }
}

#[derive(Clone)]
pub struct StopWordFilterWrapper<T> {
    words: Arc<HashSet<String>>,
    inner: T,
}

impl<T: Tokenizer> Tokenizer for StopWordFilterWrapper<T> {
    type TokenStream<'a> = StopWordFilterStream<T::TokenStream<'a>>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        StopWordFilterStream {
            words: self.words.clone(),
            tail: self.inner.token_stream(text),
        }
    }
}

pub struct StopWordFilterStream<T> {
    words: Arc<HashSet<String>>,
    tail: T,
}

impl<T> StopWordFilterStream<T> {
    fn predicate(&self, token: &Token) -> bool {
        !self.words.contains(&token.text)
    }
}

impl<T: TokenStream> TokenStream for StopWordFilterStream<T> {
    fn advance(&mut self) -> bool {
        while self.tail.advance() {
            if self.predicate(self.tail.token()) {
                return true;
            }
        }
        false
    }

    fn token(&self) -> &Token {
        self.tail.token()
    }

    fn token_mut(&mut self) -> &mut Token {
        self.tail.token_mut()
    }
}
