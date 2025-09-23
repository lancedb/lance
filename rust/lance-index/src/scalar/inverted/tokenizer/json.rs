// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{Array, ArrayRef, LargeBinaryArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream};
use futures::Stream;
use log::debug;
use serde_json::Value;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tantivy::tokenizer::{SimpleTokenizer, Token, TokenStream, Tokenizer};

/// Tokenizer for json triplet(utf8) text.
///
/// One json triplet is a triplet in format "field_name,field_type,field_value", multiples triplets
/// are separated by a semicolon.
///
/// Example:
/// text: "title,str,harrypotter;title,str,chapter;title,str,one"
/// triplets:
///   - "title,str,harrypotter"
///   - "title,str,chapter"
///   - "title,str,one"
#[derive(Clone)]
pub struct TripletTokenizer;

impl Tokenizer for TripletTokenizer {
    type TokenStream<'a> = TTStream;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> TTStream {
        let mut tokens = Vec::new();
        let mut byte_offset = 0;
        let mut idx = 0;

        for triple in text.split(';') {
            let parts: Vec<&str> = triple.splitn(3, ',').collect();
            if parts.len() != 3 {
                debug!("Invalid triple format: {}", triple);
                byte_offset += triple.len() + 1;
                continue;
            }

            let start = byte_offset;
            let end = start + triple.len();
            tokens.push(Token {
                offset_from: start,
                offset_to: end,
                position: idx,
                text: triple.to_string(),
                position_length: 1,
            });
            byte_offset += triple.len() + 1;
            idx += 1;
        }

        TTStream { tokens, index: 0 }
    }
}

pub struct TTStream {
    tokens: Vec<Token>,
    index: usize,
}

impl TokenStream for TTStream {
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

/// Transform jsonb stream into json triplet(utf8) text stream
pub struct JsonTripletStream {
    inner: SendableRecordBatchStream,
    jsonb_col: String,
}

impl JsonTripletStream {
    pub fn new(inner: SendableRecordBatchStream, jsonb_col: String) -> Self {
        Self { inner, jsonb_col }
    }
}

impl Stream for JsonTripletStream {
    type Item = datafusion_common::Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                let cols: Vec<ArrayRef> = batch
                    .schema()
                    .fields()
                    .iter()
                    .enumerate()
                    .map(|(idx, col)| {
                        if col.name().as_str() == self.jsonb_col {
                            Ok(jsonb_to_json(batch.column(idx), &self.jsonb_col)?)
                        } else {
                            Ok(batch.column(idx).clone())
                        }
                    })
                    .collect::<lance_core::Result<Vec<ArrayRef>>>()?;

                let new_schema = batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|col| {
                        if col.name().as_str() == self.jsonb_col {
                            Field::new(&self.jsonb_col, DataType::LargeUtf8, true)
                        } else {
                            col.as_ref().clone()
                        }
                    })
                    .collect::<Vec<Field>>();
                let new_schema = Arc::new(Schema::new(new_schema));
                let mapped = RecordBatch::try_new(new_schema, cols).unwrap();
                Poll::Ready(Some(Ok(mapped)))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl RecordBatchStream for JsonTripletStream {
    fn schema(&self) -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new(
            &self.jsonb_col,
            DataType::Utf8,
            true,
        )]))
    }
}

pub fn jsonb_to_json(col: &ArrayRef, col_name: &str) -> lance_core::Result<ArrayRef> {
    let binary_array = col
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .unwrap_or_else(|| panic!("column {} is not a large binary array", col_name));
    let mut builder =
        arrow_array::builder::LargeStringBuilder::with_capacity(binary_array.len(), 1024);
    for i in 0..binary_array.len() {
        if binary_array.is_null(i) {
            builder.append_null();
        } else if let Some(bytes) = binary_array.value(i).into() {
            let text = flatten_jsonb(bytes)?;
            builder.append_value(text);
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn flatten_jsonb(bytes: &[u8]) -> lance_core::Result<String> {
    let raw_jsonb = jsonb::RawJsonb::new(bytes);
    let value: Value = serde_json::from_slice(raw_jsonb.to_string().as_bytes())?;
    let mut tokens = vec![];
    let mut tokenizer = SimpleTokenizer::default();
    flatten_json(&value, "", &mut tokens, &mut tokenizer);
    let text = tokens
        .into_iter()
        .map(|(path, v_type, value)| format!("{},{},{}", path, v_type, value))
        .collect::<Vec<String>>()
        .join(";");
    Ok(text)
}

fn flatten_json(
    value: &Value,
    prefix: &str,
    out: &mut Vec<(String, String, String)>,
    tokenizer: &mut SimpleTokenizer,
) {
    match value {
        Value::Object(map) => {
            for (k, v) in map {
                let next_prefix = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{}.{}", prefix, k)
                };
                flatten_json(v, &next_prefix, out, tokenizer);
            }
        }
        Value::Array(arr) => {
            for v in arr.iter() {
                flatten_json(v, prefix, out, tokenizer);
            }
        }
        Value::String(text) => {
            let mut tokens = tokenizer.token_stream(text);
            while let Some(token) = tokens.next() {
                out.push((prefix.to_string(), "str".to_string(), token.text.clone()));
            }
        }
        _ => {
            let value_type = match value {
                Value::Null => "null",
                Value::Bool(_) => "bool",
                Value::Number(_) => "number",
                _ => unreachable!(),
            };
            out.push((
                prefix.to_string(),
                value_type.to_string(),
                value.to_string(),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::scalar::inverted::json::{flatten_json, flatten_jsonb};
    use crate::scalar::inverted::tokenizer::json::TripletTokenizer;
    use serde_json::Value;
    use tantivy::tokenizer::{SimpleTokenizer, Token, TokenStream, Tokenizer};

    #[test]
    fn test_json_tokenizer() {
        let text = r#"a,number,1;b.c,str,d;b.c,str,e"#;
        let mut tokenizer = TripletTokenizer {};
        let mut stream = tokenizer.token_stream(text);

        let mut tokens: Vec<Token> = vec![];
        while let Some(token) = stream.next() {
            tokens.push(token.clone());
        }

        assert_eq!(tokens.len(), 3);
        assert_eq!(
            tokens[0],
            Token {
                offset_from: 0,
                offset_to: 10,
                position: 0,
                text: "a,number,1".to_string(),
                position_length: 1,
            }
        );
        assert_eq!(
            tokens[1],
            Token {
                offset_from: 11,
                offset_to: 20,
                position: 1,
                text: "b.c,str,d".to_string(),
                position_length: 1,
            }
        );
        assert_eq!(
            tokens[2],
            Token {
                offset_from: 21,
                offset_to: 30,
                position: 2,
                text: "b.c,str,e".to_string(),
                position_length: 1,
            }
        );
    }

    fn flatten_json_value(value: &Value) -> lance_core::Result<String> {
        let mut tokens = vec![];
        let mut tokenizer = SimpleTokenizer::default();
        flatten_json(value, "", &mut tokens, &mut tokenizer);
        let text = tokens
            .into_iter()
            .map(|(path, v_type, value)| format!("{},{},{}", path, v_type, value))
            .collect::<Vec<String>>()
            .join(";");
        Ok(text)
    }

    #[test]
    fn test_flatten_json_text() {
        let json = r#"{
              "a": 1,
              "b": [
                {"c": "hello world"},
                {"c": "e"}
              ],
              "c": true,
              "d": null,
              "e": {
                "f": 1.0
              }
          }"#;
        let value: Value = serde_json::from_str(json).unwrap();
        let text = flatten_json_value(&value).unwrap();
        assert_eq!(
            text.as_str(),
            "a,number,1;b.c,str,hello;b.c,str,world;b.c,str,e;c,bool,true;d,null,null;e.f,number,1.0"
        );

        let json = r#"{}"#;
        let value: Value = serde_json::from_str(json).unwrap();
        let text = flatten_json_value(&value).unwrap();
        assert_eq!(text.as_str(), "");
    }

    #[test]
    fn test_flatten_jsonb() {
        let json = r#"{"a": [1, 2, 3]}"#;
        let jsonb_bytes = jsonb::parse_value(json.as_bytes()).unwrap().to_vec();
        let json_bytes: &[u8] = &jsonb_bytes;
        let text = flatten_jsonb(json_bytes).unwrap();
        assert_eq!(text.as_str(), "a,number,1;a,number,2;a,number,3");
    }
}
