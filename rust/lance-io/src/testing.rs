use std::fmt::{self, Display, Formatter};

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use mockall::mock;
use object_store::{
    path::Path, GetOptions, GetResult, ListResult, MultipartId, ObjectMeta,
    ObjectStore as OSObjectStore, PutOptions, PutResult, Result as OSResult,
};
use std::future::Future;
use tokio::io::AsyncWrite;

mock! {
    pub ObjectStore {}

    #[async_trait]
    impl OSObjectStore for ObjectStore {
        async fn put_opts(&self, location: &Path, bytes: Bytes, opts: PutOptions) -> OSResult<PutResult>;
        async fn put_multipart(
            &self,
            location: &Path,
        ) -> OSResult<(MultipartId, Box<dyn AsyncWrite + Unpin + Send>)>;
        async fn abort_multipart(&self, location: &Path, multipart_id: &MultipartId) -> OSResult<()>;
        fn get_opts<'life0, 'life1, 'async_trait>(
            &'life0 self,
            location: &'life1 Path,
            options: GetOptions
        ) -> std::pin::Pin<Box<dyn Future<Output=OSResult<GetResult> > +Send+'async_trait> > where
        Self: 'async_trait,
        'life0: 'async_trait,
        'life1: 'async_trait;
        async fn delete(&self, location: &Path) -> OSResult<()>;
        fn list<'a>(&'a self, prefix: Option<&'a Path>) -> BoxStream<'_, OSResult<ObjectMeta>>;
        async fn list_with_delimiter<'a, 'b>(&'a self, prefix: Option<&'b Path>) -> OSResult<ListResult>;
        async fn copy(&self, from: &Path, to: &Path) -> OSResult<()>;
        async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> OSResult<()>;
    }
}

impl std::fmt::Debug for MockObjectStore {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "MockObjectStore")
    }
}

impl Display for MockObjectStore {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "MockObjectStore")
    }
}
