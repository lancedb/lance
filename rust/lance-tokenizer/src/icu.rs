// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use icu_segmenter::{WordSegmenter, WordSegmenterBorrowed, options::WordBreakInvariantOptions};

use crate::{TextAnalyzer, TextAnalyzerBuilder, Token, TokenStream, Tokenizer};

#[derive(Clone)]
pub struct IcuTokenizer {
    segmenter: WordSegmenterBorrowed<'static>,
}

impl Default for IcuTokenizer {
    fn default() -> Self {
        Self {
            segmenter: WordSegmenter::new_dictionary(WordBreakInvariantOptions::default()),
        }
    }
}

impl IcuTokenizer {
    pub fn analyzer(self) -> TextAnalyzer {
        TextAnalyzer::builder(self).build()
    }

    pub fn analyzer_builder(self) -> TextAnalyzerBuilder {
        TextAnalyzer::builder(self).dynamic()
    }
}

pub struct IcuTokenStream {
    tokens: Vec<Token>,
    index: usize,
}

impl TokenStream for IcuTokenStream {
    fn advance(&mut self) -> bool {
        if self.index < self.tokens.len() {
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn token(&self) -> &Token {
        &self.tokens[self.index - 1]
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.tokens[self.index - 1]
    }
}

impl Tokenizer for IcuTokenizer {
    type TokenStream<'a> = IcuTokenStream;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        let mut boundaries = self.segmenter.segment_str(text);
        let mut tokens = Vec::new();
        let Some(mut offset_from) = boundaries.next() else {
            return IcuTokenStream { tokens, index: 0 };
        };

        for offset_to in boundaries {
            let token_text = &text[offset_from..offset_to];
            if token_text.chars().any(char::is_alphanumeric) {
                tokens.push(Token {
                    offset_from,
                    offset_to,
                    position: tokens.len(),
                    text: token_text.to_owned(),
                    position_length: 1,
                });
            }
            offset_from = offset_to;
        }

        IcuTokenStream { tokens, index: 0 }
    }
}

#[cfg(test)]
mod tests {
    use crate::{IcuTokenizer, Token, TokenStream, Tokenizer};

    fn collect_tokens(text: &str) -> Vec<Token> {
        let mut tokenizer = IcuTokenizer::default();
        let mut stream = tokenizer.token_stream(text);
        let mut tokens = Vec::new();
        stream.process(&mut |token| tokens.push(token.clone()));
        tokens
    }

    #[test]
    fn test_icu_tokenizer_segments_mixed_text() {
        let tokens = collect_tokens("Hello, こんにちは世界!");

        assert_eq!(
            tokens
                .iter()
                .map(|token| token.text.as_str())
                .collect::<Vec<_>>(),
            vec!["Hello", "こんにちは", "世界"]
        );
        assert_eq!(
            tokens
                .iter()
                .map(|token| (token.offset_from, token.offset_to, token.position))
                .collect::<Vec<_>>(),
            vec![(0, 5, 0), (7, 22, 1), (22, 28, 2)]
        );
    }

    #[test]
    fn test_icu_tokenizer_skips_non_word_segments() {
        let tokens = collect_tokens("Mark'd ye his words?");

        assert_eq!(
            tokens
                .iter()
                .map(|token| token.text.as_str())
                .collect::<Vec<_>>(),
            vec!["Mark'd", "ye", "his", "words"]
        );
    }
}
