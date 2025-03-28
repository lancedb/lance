// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

// Inspired by AmazonS3ConfigKey and friends from object-store

use std::str::FromStr;

/// Configuration keys for Lance Object Store
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum LanceStorageOption {
    /// Whether to use to use the same size for all parts in a multipart upload.
    /// By default, this is false, unless `ENDPOINT_URL` is set to a Cloudflare
    /// R2 endpoint.
    UseConstantSizeUploadParts,
    /// Whether it is safe to assume that list operations return results in
    /// lexicographical order. This is used for optimizing discover of the
    /// latest manifest.
    ListIsLexigraphicallySorted,
    /// The number of IO operations to perform in parallel.
    IoParallelism,
}

impl AsRef<str> for LanceStorageOption {
    fn as_ref(&self) -> &str {
        match self {
            Self::UseConstantSizeUploadParts => "lance_use_constant_size_upload_parts",
            Self::ListIsLexigraphicallySorted => "lance_list_is_lexigraphically_sorted",
            Self::IoParallelism => "lance_io_parallelism",
        }
    }
}

impl FromStr for LanceStorageOption {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "lance_use_constant_size_upload_parts" => Ok(Self::UseConstantSizeUploadParts),
            "lance_list_is_lexigraphically_sorted" => Ok(Self::ListIsLexigraphicallySorted),
            "lance_io_parallelism" => Ok(Self::IoParallelism),
            _ => Err(()),
        }
    }
}

fn extract_lance_storage_options<'a>(
    options: impl IntoIterator<Item = (&'a str, &'a str)> + 'a,
) -> impl Iterator<Item = (LanceStorageOption, &'a str)> + 'a {
    options.into_iter().filter_map(|(key, value)| {
        let key = key.parse().ok()?;
        Some((key, value))
    })
}

#[derive(Default, Debug)]
pub struct LanceStorageConfig {
    use_constant_size_upload_parts: Option<bool>,
    list_is_lexigraphically_sorted: Option<bool>,
    io_parallelism: Option<usize>,
}

impl LanceStorageConfig {
    fn with_config(&mut self, key: LanceStorageOption, value: &str) {
        match key {
            LanceStorageOption::UseConstantSizeUploadParts => {
                self.use_constant_size_upload_parts = Some(value.parse().unwrap());
            }
            LanceStorageOption::ListIsLexigraphicallySorted => {
                self.list_is_lexigraphically_sorted = Some(value.parse().unwrap());
            }
            LanceStorageOption::IoParallelism => {
                self.io_parallelism = Some(value.parse().unwrap());
            }
        }
    }
}

pub fn infer_lance_storage_options<'a>(
    options: impl IntoIterator<Item = (&'a str, &'a str)> + 'a,
) -> LanceStorageConfig {
    let mut config = LanceStorageConfig::default();

    for (os_key, os_value) in std::env::vars_os() {
        if let (Some(key), Some(value)) = (os_key.to_str(), os_value.to_str()) {
            if key.starts_with("LANCE_") {
                if let Ok(config_key) = key.to_ascii_lowercase().parse() {
                    config.with_config(config_key, value);
                }
            }
        }
    }

    for (key, value) in extract_lance_storage_options(options) {
        config.with_config(key, value);
    }

    config
}
