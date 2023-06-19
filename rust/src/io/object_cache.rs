use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;

use super::object_reader::ObjectReader;

#[async_trait]
pub trait ObjectCache: Send + Sync + Debug + 'static {
    async fn to_cached_reader(&self, reader: Box<dyn ObjectReader>) -> Box<dyn ObjectReader>;
}

#[derive(Debug)]
struct NoOpCache;

#[async_trait]
impl ObjectCache for NoOpCache {
    async fn to_cached_reader(&self, reader: Box<dyn ObjectReader>) -> Box<dyn ObjectReader> {
        reader
    }
}

pub struct ObjectCaches;

impl ObjectCaches {
    pub fn no_op() -> Arc<dyn ObjectCache> {
        Arc::new(NoOpCache)
    }
}
