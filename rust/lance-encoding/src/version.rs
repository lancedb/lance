// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::{Error, Result};
use snafu::{location, Location};

pub const LEGACY_FORMAT_VERSION: &str = "0.1";
pub const V2_FORMAT_2_0: &str = "2.0";
pub const V2_FORMAT_2_1: &str = "2.1";

/// Lance file version
#[derive(Debug, PartialEq, Eq, Clone, Copy, Ord, PartialOrd)]
pub enum LanceFileVersion {
    // Note that Stable must come AFTER the stable version and Next must come AFTER the next version
    // this way comparisons like x >= V2_0 will work the same if x is Stable or V2_0
    /// The legacy (0.1) format
    Legacy,
    V2_0,
    /// The latest stable release
    Stable,
    V2_1,
    /// The latest unstable release
    Next,
}

impl Default for LanceFileVersion {
    fn default() -> Self {
        // Changing this is impactful beyond lance_file, this is the default used for new datasets in Lance
        LanceFileVersion::Legacy
    }
}

impl LanceFileVersion {
    /// Convert Stable or Next to the actual version
    pub fn resolve(&self) -> LanceFileVersion {
        match self {
            LanceFileVersion::Legacy => LanceFileVersion::Legacy,
            LanceFileVersion::V2_0 => LanceFileVersion::V2_0,
            LanceFileVersion::V2_1 => LanceFileVersion::V2_1,
            LanceFileVersion::Stable => LanceFileVersion::V2_0,
            LanceFileVersion::Next => LanceFileVersion::V2_1,
        }
    }

    /// Returns the default version if Legacy is not an option
    pub fn default_v2() -> Self {
        // This will go away soon, but there are a few spots where the Legacy default doesn't make sense
        LanceFileVersion::V2_0
    }
}

impl std::fmt::Display for LanceFileVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", String::from(*self))
    }
}

impl From<LanceFileVersion> for String {
    fn from(version: LanceFileVersion) -> Self {
        match version {
            LanceFileVersion::Legacy => LEGACY_FORMAT_VERSION.to_string(),
            LanceFileVersion::V2_0 => V2_FORMAT_2_0.to_string(),
            LanceFileVersion::V2_1 => V2_FORMAT_2_1.to_string(),
            LanceFileVersion::Stable => "stable".to_string(),
            LanceFileVersion::Next => "next".to_string(),
        }
    }
}

impl TryFrom<&str> for LanceFileVersion {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self> {
        match value.to_lowercase().as_str() {
            LEGACY_FORMAT_VERSION => Ok(Self::Legacy),
            V2_FORMAT_2_0 => Ok(Self::V2_0),
            V2_FORMAT_2_1 => Ok(Self::V2_1),
            "stable" => Ok(Self::Stable),
            "legacy" => Ok(Self::Legacy),
            // Version 0.3 is an alias of 2.0
            "0.3" => Ok(Self::V2_0),
            _ => Err(Error::InvalidInput {
                source: format!("Unknown Lance storage version: {}", value).into(),
                location: location!(),
            }),
        }
    }
}

impl TryFrom<String> for LanceFileVersion {
    type Error = Error;

    fn try_from(value: String) -> Result<Self> {
        Self::try_from(value.as_str())
    }
}
