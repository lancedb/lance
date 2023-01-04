use std::io::Result;

use object_store::{memory::InMemory, path::Path, ObjectStore};

use self::scanner::Scanner;

mod scanner;

/// Lance Dataset
#[derive(Debug)]
pub struct Dataset {
    object_store: Box<dyn ObjectStore>,
    base: Path,
}

impl Dataset {
    pub fn open(uri: &str) -> Result<Self> {
        Ok(Self {
            object_store: Box::new(InMemory::new()),
            base: Path::from(uri),
        })
    }

    pub fn scan(&self) -> Result<Scanner> {
        todo!()
    }

    pub fn object_store(&self) -> &dyn ObjectStore {
        self.object_store.as_ref()
    }
}
