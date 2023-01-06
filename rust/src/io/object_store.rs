//! Wraps [ObjectStore](object_store::ObjectStore)
use std::sync::Arc;
use std::io::{Error, Result, ErrorKind};

use ::object_store::{
    aws::AmazonS3Builder, memory::InMemory, path::Path,
    ObjectStore as OSObjectStore,
};
use url::Url;

use super::object_reader::ObjectReader;

/// Wraps [ObjectStore](object_store::ObjectStore)
#[derive(Debug)]
pub struct ObjectStore {
    // Inner object store
    pub inner: Arc<dyn OSObjectStore>,
    scheme: String,
    bucket: String,
    base_path: Path,
    prefetch_size: usize,
}

impl std::fmt::Display for ObjectStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ObjectStore({})", self.scheme)
    }
}

impl ObjectStore {
    pub fn new(uri: &str) -> Result<Self> {
        if uri == ":memory:" {
            return Ok(Self {
                inner: Arc::new(InMemory::new()),
                scheme: String::from("memory"),
                bucket: String::from(""),
                base_path: Path::from(""),
                prefetch_size: 64 * 1024,
            });
        };

        let parsed = match Url::parse(uri) {
            Ok(u) => u,
            Err(_e) => {
                return Err(Error::new(ErrorKind::InvalidInput, "Invalid uri "));
            }
        };

        let _bucket_name = parsed.host().unwrap().to_string();
        let _object_store = match parsed.scheme() {
            "s3" => {
                let bucket_name = parsed.host().unwrap().to_string();

                match AmazonS3Builder::from_env()
                    .with_bucket_name(bucket_name)
                    .build()
                {
                    Ok(s3) => Arc::new(s3),
                    Err(e) => return Err(e.into()),
                }
            }
            &_ => todo!(),
        };

        todo!()
    }

    pub fn prefetch_size(&self) -> usize {
        self.prefetch_size
    }

    pub fn set_prefetch_size(&mut self, new_size: usize) {
        self.prefetch_size = new_size;
    }

    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    pub async fn open(&self, path: &Path) -> Result<ObjectReader> {
        match ObjectReader::new(self.inner.clone(), path.clone(), self.prefetch_size) {
            Ok(r) => Ok(r),
            Err(e) => Err(e.into()),
        }
    }
}

