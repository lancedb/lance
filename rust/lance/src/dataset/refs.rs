use lance_table::format::pb;

/// Nice names for dataset versions
///
/// Used to apply higher-order abstractions over versions such as tags and 
/// branches.
#[derive(Debug, Clone)]
pub struct Refs {
    pub tags: HashMap::<String, u64>,
    pub heads: HashMap::<String, u64>,
}

impl Refs {
    pub fn new() -> Self {
        Self {
            tags: HashMap::<String, u64>::new(),
            heads: HashMap::<String, u64>::new(),
        }
    }
}

impl TryFrom<pb::Refs> for Refs {
    type Error = Error;

    fn try_from(message: pb::Refs) -> Result<Self> {
        Ok(Self {
            tags: message.tags.clone(),
            heads: message.heads.clone(),
        })
    }
}

impl From<&Refs> for pb::Refs {
    fn from(value: &Refs) -> Self {
        Self {
            tags: value.tags,
            heads: value.heads,
        }
    }
}
