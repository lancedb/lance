use serde::{Deserialize, Serialize};

pub type Tags = Vec<Tag>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tag {
    pub name: String,
    pub version: u64,
    pub manifest_size: u64,
}
