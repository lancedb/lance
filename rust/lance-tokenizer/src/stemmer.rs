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
            Self::Greek => StemmerAlgorithm::Greek,
            language => StemmerAlgorithm::Legacy(language.legacy_algorithm()),
        }
    }

    fn legacy_algorithm(self) -> rust_stemmers::Algorithm {
        match self {
            Self::Arabic => rust_stemmers::Algorithm::Arabic,
            Self::Danish => rust_stemmers::Algorithm::Danish,
            Self::Dutch => rust_stemmers::Algorithm::Dutch,
            Self::English => rust_stemmers::Algorithm::English,
            Self::Finnish => rust_stemmers::Algorithm::Finnish,
            Self::French => rust_stemmers::Algorithm::French,
            Self::German => rust_stemmers::Algorithm::German,
            Self::Greek => rust_stemmers::Algorithm::Greek,
            Self::Hungarian => rust_stemmers::Algorithm::Hungarian,
            Self::Italian => rust_stemmers::Algorithm::Italian,
            Self::Norwegian => rust_stemmers::Algorithm::Norwegian,
            Self::Portuguese => rust_stemmers::Algorithm::Portuguese,
            Self::Romanian => rust_stemmers::Algorithm::Romanian,
            Self::Russian => rust_stemmers::Algorithm::Russian,
            Self::Spanish => rust_stemmers::Algorithm::Spanish,
            Self::Swedish => rust_stemmers::Algorithm::Swedish,
            Self::Tamil => rust_stemmers::Algorithm::Tamil,
            Self::Turkish => rust_stemmers::Algorithm::Turkish,
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
            Self::Greek => StemmerBackend::Greek(frostem::Stemmer::new(frostem::Algorithm::Greek)),
        }
    }
}

enum StemmerBackend {
    Legacy(rust_stemmers::Stemmer),
    Greek(frostem::Stemmer),
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

    /// Create a stemmer with the semantics used by indexes written before the
    /// corrected Greek stemmer was introduced.
    ///
    /// This is only intended for reading and incrementally updating persisted
    /// index metadata that does not identify its Greek stemmer version.
    #[doc(hidden)]
    pub fn new_legacy(language: Language) -> Self {
        Self {
            stemmer_algorithm: StemmerAlgorithm::Legacy(language.legacy_algorithm()),
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

    #[test]
    fn test_legacy_greek_stemmer_preserves_existing_terms() {
        let mut legacy = TextAnalyzer::builder(RawTokenizer::default())
            .filter(Stemmer::new_legacy(Language::Greek))
            .build();
        let mut current = TextAnalyzer::builder(RawTokenizer::default())
            .filter(Stemmer::new(Language::Greek))
            .build();

        let mut legacy_stream = legacy.token_stream("ίσα");
        assert!(legacy_stream.advance());
        assert_eq!(legacy_stream.token().text, "");

        let mut current_stream = current.token_stream("ίσα");
        assert!(current_stream.advance());
        assert_eq!(current_stream.token().text, "ισ");
    }
}
