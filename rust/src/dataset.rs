//! Lance Dataset
//!

use std::collections::BTreeMap;
use std::io::Result;

use chrono::prelude::*;
use object_store::path::Path;

use self::scanner::Scanner;
use crate::datatypes::Schema;
use crate::format::{pb, Manifest};
use crate::io::reader::read_manifest;
use crate::io::{read_metadata_offset, ObjectStore};

pub mod scanner;

const LATEST_MANIFEST_NAME: &str = "_latest.manifest";
const VERSIONS_DIR: &str = "_versions";
const DATA_DIR: &str = "data";

fn latest_manifest_path(base: &Path) -> Path {
    base.child(LATEST_MANIFEST_NAME)
}

/// Lance Dataset
#[derive(Debug)]
pub struct Dataset {
    object_store: ObjectStore,
    base: Path,
    manifest: Manifest,
}

/// Dataset Version
pub struct Version {
    /// version number
    pub version: u64,

    /// Timestamp of dataset creation in UTC.
    pub timestamp: DateTime<Utc>,

    /// Key-value pairs of metadata.
    pub metadata: BTreeMap<String, String>,
}

/// Convert Manifest to Data Version.
impl From<&Manifest> for Version {
    fn from(m: &Manifest) -> Self {
        Version {
            version: m.version,
            timestamp: Utc::now(),
            metadata: BTreeMap::default(),
        }
    }
}

impl Dataset {
    /// Open an existing dataset.
    pub async fn open(uri: &str) -> Result<Self> {
        let object_store = ObjectStore::new(uri)?;

        let latest_manifest_path = latest_manifest_path(object_store.base_path());

        let mut object_reader = object_store.open(&latest_manifest_path).await?;
        let bytes = object_store
            .inner
            .get(&latest_manifest_path)
            .await?
            .bytes()
            .await?;
        let offset = read_metadata_offset(&bytes)?;
        let manifest_pb = object_reader.read_message::<pb::Manifest>(offset).await?;
        let manifest = (&manifest_pb).into();

        Ok(Self {
            object_store,
            base: Path::from(uri),
            manifest: manifest,
        })
    }

    pub fn scan(&self) -> Result<Scanner> {
        todo!()
    }

    pub fn object_store(&self) -> &ObjectStore {
        &self.object_store
    }

    fn versions_dir(&self) -> Path {
        self.base.child(VERSIONS_DIR)
    }

    pub fn version(&self) -> Version {
        Version::from(&self.manifest)
    }

    /// Get all versions.
    pub async fn versions(&self) -> Result<Vec<Version>> {
        let paths: Vec<Path> = self
            .object_store
            .inner
            .list_with_delimiter(Some(&self.versions_dir()))
            .await?
            .objects
            .iter()
            .filter(|&obj| obj.location.as_ref().ends_with(".manifest"))
            .map(|o| o.location.clone())
            .collect();
        let mut versions = vec![];
        for path in paths.iter() {
            let manifest = read_manifest(&self.object_store, path).await?;
            versions.push(Version::from(&manifest));
        }
        Ok(versions)
    }

    pub fn schema(&self) -> &Schema {
        &self.manifest.schema
    }
}
