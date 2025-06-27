// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::path::{Path, PathBuf};

use lance_core::{Error, Result};
use lindera_tantivy::tokenizer::LinderaTokenizer;
use snafu::location;

pub const LINDERA_LANGUAGE_MODEL_CONFIG_FILE: &str = "config.yml";

pub trait LinderaTokenizerBuilder: Sized {
    fn load(p: &Path) -> Result<Self> {
        if !p.is_dir() {
            return Err(Error::io(
                format!("{} is not a valid directory", p.display()),
                snafu::location!(),
            ));
        }
        let config_path = p.join(LINDERA_LANGUAGE_MODEL_CONFIG_FILE);
        Self::new(config_path.as_path())
    }

    fn new(config_path: &Path) -> Result<Self>;

    fn build(&self) -> Result<tantivy::tokenizer::TextAnalyzerBuilder>;
}

pub struct LinderaBuilder {
    config_path: PathBuf,
}

impl LinderaTokenizerBuilder for LinderaBuilder {
    fn new(config_path: &Path) -> Result<Self> {
        Ok(Self {
            config_path: config_path.to_path_buf(),
        })
    }

    fn build(&self) -> Result<tantivy::tokenizer::TextAnalyzerBuilder> {
        let tokenizer = if self.config_path.exists() {
            match LinderaTokenizer::from_file(&self.config_path) {
                Ok(tok) => tok,
                Err(e) => {
                    return Err(Error::io(
                        format!(
                            "Failed to load tokenizer config at {}: {}",
                            self.config_path.display(),
                            e
                        ),
                        location!(),
                    ));
                }
            }
        } else {
            log::info!(
                "Config file not found at '{}'. Falling back to `LINDERA_CONFIG_PATH`.",
                self.config_path.display(),
            );
            LinderaTokenizer::new().map_err(|e| {
                Error::io(
                    format!("Failed to initialize default tokenizer: {e}"),
                    location!(),
                )
            })?
        };
        Ok(tantivy::tokenizer::TextAnalyzer::builder(tokenizer).dynamic())
    }
}
