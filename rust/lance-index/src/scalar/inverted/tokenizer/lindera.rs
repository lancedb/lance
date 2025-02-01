// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::path::{Path, PathBuf};

use super::TokenizerBuilder;
use lance_core::{Error, Result};
use lindera::{
    dictionary::{
        load_dictionary_from_path, load_user_dictionary_from_config, UserDictionaryConfig,
    },
    mode::Mode,
    segmenter::Segmenter,
};
use lindera_tantivy::tokenizer::LinderaTokenizer;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use snafu::location;

#[derive(Serialize, Deserialize, Default)]
pub struct LinderaConfig {
    main: Option<String>,
    user: Option<String>,
    user_kind: Option<String>,
}

pub struct LinderaBuilder {
    root: PathBuf,
    config: LinderaConfig,
}

impl LinderaBuilder {
    fn main_dict_path(&self) -> PathBuf {
        if let Some(p) = &self.config.main {
            return self.root.join(p);
        }
        self.root.join("main")
    }

    fn user_dict_config(&self) -> Result<Option<UserDictionaryConfig>> {
        let Some(user_dict_path) = &self.config.user else {
            return Ok(None);
        };
        let mut conf = Map::<String, Value>::new();
        let user_path = self.root.join(user_dict_path);
        let Some(p) = user_path.to_str() else {
            return Err(Error::io(
                format!(
                    "invalid lindera tokenizer user dictionary path: {}",
                    user_path.display()
                ),
                location!(),
            ));
        };
        conf.insert(String::from("path"), Value::String(String::from(p)));
        if let Some(kind) = &self.config.user_kind {
            conf.insert(String::from("kind"), Value::String(kind.clone()));
        }
        Ok(Some(Value::Object(conf)))
    }
}

impl TokenizerBuilder for LinderaBuilder {
    type Config = LinderaConfig;

    fn new(config: Self::Config, root: &Path) -> Result<Self> {
        Ok(Self {
            config,
            root: root.to_path_buf(),
        })
    }

    fn build(&self) -> Result<tantivy::tokenizer::TextAnalyzerBuilder> {
        let main_path = self.main_dict_path();
        let dictionary = load_dictionary_from_path(main_path.as_path()).map_err(|e| {
            Error::io(
                format!(
                    "load lindera tokenizer main dictionary from {}, error: {}",
                    main_path.display(),
                    e
                ),
                location!(),
            )
        })?;
        let user_dictionary = match self.user_dict_config()? {
            Some(conf) => {
                let user_dictionary = load_user_dictionary_from_config(&conf).map_err(|e| {
                    Error::io(
                        format!("load lindera tokenizer user dictionary, conf:{conf}, err: {e}"),
                        location!(),
                    )
                })?;
                Some(user_dictionary)
            }
            None => None,
        };
        let mode = Mode::Normal;
        let segmenter = Segmenter::new(mode, dictionary, user_dictionary);
        let tokenizer = LinderaTokenizer::from_segmenter(segmenter);
        Ok(tantivy::tokenizer::TextAnalyzer::builder(tokenizer).dynamic())
    }
}
