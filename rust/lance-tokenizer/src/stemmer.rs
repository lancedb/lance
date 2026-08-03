// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-License-Identifier: MIT
// Adapted from Tantivy v0.24.2 stemmer filter.
// Copyright (c) 2017-present Tantivy contributors.

use std::borrow::Cow;
use std::mem;

use serde::{Deserialize, Serialize};

use crate::{Token, TokenFilter, TokenStream, Tokenizer};

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Copy, Clone)]
pub enum Language {
    Arabic,
    Danish,
    Dutch,
    English,
    Finnish,
    French,
    German,
    Greek,
    Hungarian,
    Italian,
    Norwegian,
    Portuguese,
    Romanian,
    Russian,
    Spanish,
    Swedish,
    Tamil,
    Turkish,
}

impl Language {
    fn algorithm(self) -> StemmerAlgorithm {
        match self {
            Self::Arabic => StemmerAlgorithm::Legacy(rust_stemmers::Algorithm::Arabic),
            Self::Danish => StemmerAlgorithm::Legacy(rust_stemmers::Algorithm::Danish),
            Self::Dutch => StemmerAlgorithm::Legacy(rust_stemmers::Algorithm::Dutch),
            Self::English => StemmerAlgorithm::Legacy(rust_stemmers::Algorithm::English),
            Self::Finnish => StemmerAlgorithm::Legacy(rust_stemmers::Algorithm::Finnish),
            Self::French => StemmerAlgorithm::Legacy(rust_stemmers::Algorithm::French),
            Self::German => StemmerAlgorithm::Legacy(rust_stemmers::Algorithm::German),
            Self::Greek => StemmerAlgorithm::Greek,
            Self::Hungarian => StemmerAlgorithm::Legacy(rust_stemmers::Algorithm::Hungarian),
            Self::Italian => StemmerAlgorithm::Legacy(rust_stemmers::Algorithm::Italian),
            Self::Norwegian => StemmerAlgorithm::Legacy(rust_stemmers::Algorithm::Norwegian),
            Self::Portuguese => StemmerAlgorithm::Legacy(rust_stemmers::Algorithm::Portuguese),
            Self::Romanian => StemmerAlgorithm::Legacy(rust_stemmers::Algorithm::Romanian),
            Self::Russian => StemmerAlgorithm::Legacy(rust_stemmers::Algorithm::Russian),
            Self::Spanish => StemmerAlgorithm::Legacy(rust_stemmers::Algorithm::Spanish),
            Self::Swedish => StemmerAlgorithm::Legacy(rust_stemmers::Algorithm::Swedish),
            Self::Tamil => StemmerAlgorithm::Legacy(rust_stemmers::Algorithm::Tamil),
            Self::Turkish => StemmerAlgorithm::Legacy(rust_stemmers::Algorithm::Turkish),
        }
    }
}

#[derive(Copy, Clone)]
enum StemmerAlgorithm {
    Legacy(rust_stemmers::Algorithm),
    // The legacy generated Greek algorithm can retain stale UTF-8 byte offsets
    // after shortening a word and panic when it slices the resulting stem.
    Greek,
}

impl StemmerAlgorithm {
    fn create(self) -> StemmerBackend {
        match self {
            Self::Legacy(algorithm) => {
                StemmerBackend::Legacy(rust_stemmers::Stemmer::create(algorithm))
            }
            Self::Greek => StemmerBackend::Greek(pagefind_stem::Stemmer::create(
                pagefind_stem::Algorithm::Greek,
            )),
        }
    }
}

enum StemmerBackend {
    Legacy(rust_stemmers::Stemmer),
    Greek(pagefind_stem::Stemmer),
}

impl StemmerBackend {
    fn stem<'a>(&self, input: &'a str) -> Cow<'a, str> {
        match self {
            Self::Legacy(stemmer) => stemmer.stem(input),
            Self::Greek(stemmer) => stemmer.stem(input),
        }
    }
}

#[derive(Clone)]
pub struct Stemmer {
    stemmer_algorithm: StemmerAlgorithm,
}

impl Stemmer {
    pub fn new(language: Language) -> Self {
        Self {
            stemmer_algorithm: language.algorithm(),
        }
    }
}

impl Default for Stemmer {
    fn default() -> Self {
        Self::new(Language::English)
    }
}

impl TokenFilter for Stemmer {
    type Tokenizer<T: Tokenizer> = StemmerFilter<T>;

    fn transform<T: Tokenizer>(self, tokenizer: T) -> Self::Tokenizer<T> {
        StemmerFilter {
            stemmer_algorithm: self.stemmer_algorithm,
            inner: tokenizer,
        }
    }
}

#[derive(Clone)]
pub struct StemmerFilter<T> {
    stemmer_algorithm: StemmerAlgorithm,
    inner: T,
}

impl<T: Tokenizer> Tokenizer for StemmerFilter<T> {
    type TokenStream<'a> = StemmerTokenStream<T::TokenStream<'a>>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        StemmerTokenStream {
            tail: self.inner.token_stream(text),
            stemmer: self.stemmer_algorithm.create(),
            buffer: String::new(),
        }
    }
}

pub struct StemmerTokenStream<T> {
    tail: T,
    stemmer: StemmerBackend,
    buffer: String,
}

impl<T: TokenStream> TokenStream for StemmerTokenStream<T> {
    fn advance(&mut self) -> bool {
        if !self.tail.advance() {
            return false;
        }
        let token = self.tail.token_mut();
        let stemmed = self.stemmer.stem(&token.text);
        match stemmed {
            Cow::Owned(stemmed) => token.text = stemmed,
            Cow::Borrowed(stemmed) => {
                self.buffer.clear();
                self.buffer.push_str(stemmed);
                mem::swap(&mut token.text, &mut self.buffer);
            }
        }
        true
    }

    fn token(&self) -> &Token {
        self.tail.token()
    }

    fn token_mut(&mut self) -> &mut Token {
        self.tail.token_mut()
    }
}

#[cfg(test)]
mod tests {
    use crate::{Language, RawTokenizer, Stemmer, TextAnalyzer, TokenStream};

    #[test]
    fn test_greek_stemmer_handles_multibyte_suffixes() {
        let mut analyzer = TextAnalyzer::builder(RawTokenizer::default())
            .filter(Stemmer::new(Language::Greek))
            .build();
        let mut stream = analyzer.token_stream("αντιθετε");

        assert!(stream.advance());
        assert_eq!(stream.token().text, "ανετ");
    }
}
