use crate::datatypes::Schema;

mod fragment;
use fragment::Fragment;

/// Protobuf definitions
#[allow(clippy::all)]
pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/lance.format.pb.rs"));
}

/// Manifest of a dataset
///
///  * Schema
///  * Version
///  * Fragments.
#[derive(Debug)]
pub struct Manifest {
    /// Dataset schema.
    pub schema: Schema,

    /// Dataset version
    pub version: u64,

    /// Fragments, the pieces to build the dataset.
    pub fragments: Vec<Fragment>,
}

impl From<&pb::Manifest> for Manifest {
    fn from(p: &pb::Manifest) -> Self {
        Self {
            schema: Schema::from(&p.fields),
            version: p.version,
            fragments: p.fragments.iter().map(Fragment::from).collect(),
        }
    }
}
