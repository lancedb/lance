// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::str::FromStr;

use lance_core::{Error, Result};
use snafu::{location, Location};

pub const LEGACY_FORMAT_VERSION: &str = "0.1";
pub const V2_FORMAT_2_0: &str = "2.0";
pub const V2_FORMAT_2_1: &str = "2.1";

/// Lance file version
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy, Ord, PartialOrd)]
pub enum LanceFileVersion {
    // Note that Stable must come AFTER the stable version and Next must come AFTER the next version
    // this way comparisons like x >= V2_0 will work the same if x is Stable or V2_0
    /// The legacy (0.1) format
    Legacy,
    #[default]
    V2_0,
    /// The latest stable release
    Stable,
    V2_1,
    /// The latest unstable release
    Next,
}

impl LanceFileVersion {
    /// Convert Stable or Next to the actual version
    pub fn resolve(&self) -> Self {
        match self {
            Self::Stable => Self::V2_0,
            Self::Next => Self::V2_1,
            _ => *self,
        }
    }

    pub fn try_from_major_minor(major: u32, minor: u32) -> Result<Self> {
        match (major, minor) {
            (0, 0) => Ok(Self::Legacy),
            (0, 1) => Ok(Self::Legacy),
            (0, 2) => Ok(Self::Legacy),
            (0, 3) => Ok(Self::V2_0),
            (2, 0) => Ok(Self::V2_0),
            (2, 1) => Ok(Self::V2_1),
            _ => Err(Error::InvalidInput {
                source: format!("Unknown Lance storage version: {}.{}", major, minor).into(),
                location: location!(),
            }),
        }
    }

    pub fn to_numbers(&self) -> (u32, u32) {
        match self {
            Self::Legacy => (0, 2),
            Self::V2_0 => (2, 0),
            Self::V2_1 => (2, 1),
            Self::Stable => self.resolve().to_numbers(),
            Self::Next => self.resolve().to_numbers(),
        }
    }
}

impl std::fmt::Display for LanceFileVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Legacy => LEGACY_FORMAT_VERSION,
                Self::V2_0 => V2_FORMAT_2_0,
                Self::V2_1 => V2_FORMAT_2_1,
                Self::Stable => "stable",
                Self::Next => "next",
            }
        )
    }
}

impl FromStr for LanceFileVersion {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
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
