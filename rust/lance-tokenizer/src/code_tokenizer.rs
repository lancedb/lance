// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{iter::Peekable, str::CharIndices};

use crate::{Token, TokenStream, Tokenizer};

/// Tokenizer for code-like text.
///
/// Identifiers are Unicode alphanumeric characters plus `_`. Other characters
/// are lexical boundaries. When operator indexing is enabled, contiguous
/// operator characters are emitted as tokens too.
#[derive(Clone, Default)]
pub struct CodeLexTokenizer {
    index_operators: bool,
    token: Token,
}

impl CodeLexTokenizer {
    pub fn new(index_operators: bool) -> Self {
        Self {
            index_operators,
            token: Token::default(),
        }
    }
}

pub struct CodeLexTokenStream<'a> {
    text: &'a str,
    chars: Peekable<CharIndices<'a>>,
    token: &'a mut Token,
    index_operators: bool,
}

impl Tokenizer for CodeLexTokenizer {
    type TokenStream<'a> = CodeLexTokenStream<'a>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        self.token.reset();
        CodeLexTokenStream {
            text,
            chars: text.char_indices().peekable(),
            token: &mut self.token,
            index_operators: self.index_operators,
        }
    }
}

fn is_identifier_char(ch: char) -> bool {
    ch == '_' || ch.is_alphanumeric()
}

fn is_operator_char(ch: char) -> bool {
    matches!(
        ch,
        '!' | '%' | '&' | '*' | '+' | '-' | '/' | ':' | '<' | '=' | '>' | '?' | '^' | '|' | '~'
    )
}

impl CodeLexTokenStream<'_> {
    fn search_token_end(&mut self, predicate: impl Fn(char) -> bool) -> usize {
        while let Some((_, ch)) = self.chars.peek() {
            if !predicate(*ch) {
                break;
            }
            self.chars.next();
        }
        self.chars
            .peek()
            .map(|(offset, _)| *offset)
            .unwrap_or(self.text.len())
    }
}

impl TokenStream for CodeLexTokenStream<'_> {
    fn advance(&mut self) -> bool {
        self.token.text.clear();
        while let Some((offset_from, ch)) = self.chars.next() {
            let token_end = if is_identifier_char(ch) {
                self.search_token_end(is_identifier_char)
            } else if self.index_operators && is_operator_char(ch) {
                self.search_token_end(is_operator_char)
            } else {
                continue;
            };

            self.token.position = self.token.position.wrapping_add(1);
            self.token.position_length = 1;
            self.token.offset_from = offset_from;
            self.token.offset_to = token_end;
            self.token.text.push_str(&self.text[offset_from..token_end]);
            return true;
        }
        false
    }

    fn token(&self) -> &Token {
        self.token
    }

    fn token_mut(&mut self) -> &mut Token {
        self.token
    }
}

#[cfg(test)]
mod tests {
    use crate::{CodeLexTokenizer, TextAnalyzer, Token};

    fn collect_tokens(text: &str, index_operators: bool) -> Vec<Token> {
        let mut analyzer = TextAnalyzer::builder(CodeLexTokenizer::new(index_operators)).build();
        let mut stream = analyzer.token_stream(text);
        let mut tokens = Vec::new();
        stream.process(&mut |token| tokens.push(token.clone()));
        tokens
    }

    #[test]
    fn test_code_lex_tokenizer_identifiers() {
        let tokens = collect_tokens("std::vector user-name parse.HTML2JSON", false);
        let texts = tokens
            .iter()
            .map(|token| token.text.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            texts,
            vec!["std", "vector", "user", "name", "parse", "HTML2JSON"]
        );
    }

    #[test]
    fn test_code_lex_tokenizer_operators() {
        let tokens = collect_tokens("a::b != c->d", true);
        let texts = tokens
            .iter()
            .map(|token| token.text.as_str())
            .collect::<Vec<_>>();
        assert_eq!(texts, vec!["a", "::", "b", "!=", "c", "->", "d"]);
    }
}
