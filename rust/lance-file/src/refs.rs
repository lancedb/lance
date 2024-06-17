use crate::format::pbrefsfile;

use lance_core::{Error, Result};
use std::collections::HashMap;

/// Nice names for dataset versions
///
/// Higher-order version abstractions such as tags and branches.
#[derive(Debug, Clone)]
pub struct Refs {
    pub tags: HashMap<String, u64>,
    pub heads: HashMap<String, u64>,
}

impl Refs {
    pub fn new() -> Self {
        Self {
            tags: HashMap::<String, u64>::new(),
            heads: HashMap::<String, u64>::new(),
        }
    }
}

impl TryFrom<pbrefsfile::Refs> for Refs {
    type Error = Error;

    fn try_from(message: pbrefsfile::Refs) -> Result<Self> {
        Ok(Self {
            tags: message.tags.clone(),
            heads: message.heads.clone(),
        })
    }
}

impl From<&Refs> for pbrefsfile::Refs {
    fn from(value: &Refs) -> Self {
        Self {
            tags: value.tags.clone(),
            heads: value.heads.clone(),
        }
    }
}
