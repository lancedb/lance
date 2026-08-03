// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::OnceLock;
use std::time::Duration;

use object_store_opendal::OpendalStore;
use opendal::Operator;
use opendal::layers::RetryLayer;

fn max_retries() -> usize {
    static MAX_RETRIES: OnceLock<usize> = OnceLock::new();
    *MAX_RETRIES.get_or_init(|| {
        std::env::var("LANCE_CONN_RESET_RETRIES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(20)
    })
}

/// Install retries beneath OpenDAL's writer so a retry reuses the same writer
/// state and buffer instead of allocating another cloud multipart part.
pub(crate) fn store_with_retry(operator: Operator) -> OpendalStore {
    let retry_layer = RetryLayer::default()
        .with_min_delay(Duration::from_secs(2))
        .with_max_delay(Duration::from_secs(8))
        .with_max_times(max_retries())
        .with_jitter();
    OpendalStore::new(operator.layer(retry_layer))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use bytes::Bytes;
    use object_store::path::Path;
    use object_store::{ObjectStoreExt, PutPayload};
    use opendal::raw::oio;
    use opendal::raw::*;
    use opendal::{
        Buffer, Builder, Capability, EntryMode, Error, ErrorKind, Metadata, OperationContext,
        Result,
    };

    use super::*;

    #[derive(Clone, Debug, Default)]
    struct TransientWriteBuilder {
        attempts: Arc<AtomicUsize>,
        bodies: Arc<Mutex<Vec<Bytes>>>,
    }

    impl Builder for TransientWriteBuilder {
        type Config = ();

        fn build(self) -> Result<impl Service> {
            Ok(TransientWriteService {
                attempts: self.attempts,
                bodies: self.bodies,
            })
        }
    }

    #[derive(Clone, Debug)]
    struct TransientWriteService {
        attempts: Arc<AtomicUsize>,
        bodies: Arc<Mutex<Vec<Bytes>>>,
    }

    impl Service for TransientWriteService {
        type Reader = ();
        type Writer = TransientWriter;
        type Lister = ();
        type Deleter = ();
        type Copier = ();

        fn info(&self) -> ServiceInfo {
            ServiceInfo::with_scheme("transient-write")
        }

        fn capability(&self) -> Capability {
            Capability {
                write: true,
                write_can_multi: true,
                ..Default::default()
            }
        }

        async fn create_dir(
            &self,
            _: &OperationContext,
            _: &str,
            _: OpCreateDir,
        ) -> Result<RpCreateDir> {
            Err(Error::new(ErrorKind::Unsupported, "create is unsupported"))
        }

        async fn stat(&self, _: &OperationContext, _: &str, _: OpStat) -> Result<RpStat> {
            Err(Error::new(ErrorKind::Unsupported, "stat is unsupported"))
        }

        fn read(&self, _: &OperationContext, _: &str, _: OpRead) -> Result<Self::Reader> {
            Err(Error::new(ErrorKind::Unsupported, "read is unsupported"))
        }

        fn write(&self, _: &OperationContext, _: &str, _: OpWrite) -> Result<Self::Writer> {
            Ok(TransientWriter {
                attempts: Arc::clone(&self.attempts),
                bodies: Arc::clone(&self.bodies),
            })
        }

        fn delete(&self, _: &OperationContext) -> Result<Self::Deleter> {
            Err(Error::new(ErrorKind::Unsupported, "delete is unsupported"))
        }

        fn list(&self, _: &OperationContext, _: &str, _: OpList) -> Result<Self::Lister> {
            Err(Error::new(ErrorKind::Unsupported, "list is unsupported"))
        }

        fn copy(
            &self,
            _: &OperationContext,
            _: &str,
            _: &str,
            _: OpCopy,
            _: OpCopier,
        ) -> Result<Self::Copier> {
            Err(Error::new(ErrorKind::Unsupported, "copy is unsupported"))
        }

        async fn rename(
            &self,
            _: &OperationContext,
            _: &str,
            _: &str,
            _: OpRename,
        ) -> Result<RpRename> {
            Err(Error::new(ErrorKind::Unsupported, "rename is unsupported"))
        }

        async fn presign(&self, _: &OperationContext, _: &str, _: OpPresign) -> Result<RpPresign> {
            Err(Error::new(ErrorKind::Unsupported, "presign is unsupported"))
        }
    }

    #[derive(Debug)]
    struct TransientWriter {
        attempts: Arc<AtomicUsize>,
        bodies: Arc<Mutex<Vec<Bytes>>>,
    }

    impl oio::Write for TransientWriter {
        async fn write(&mut self, body: Buffer) -> Result<()> {
            self.bodies.lock().unwrap().push(body.to_bytes());
            if self.attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                Err(Error::new(ErrorKind::Unexpected, "connection reset by peer").set_temporary())
            } else {
                Ok(())
            }
        }

        async fn close(&mut self) -> Result<Metadata> {
            Ok(Metadata::new(EntryMode::FILE))
        }

        async fn abort(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test(start_paused = true)]
    async fn test_retry_reuses_opendal_writer_and_body() {
        let builder = TransientWriteBuilder::default();
        let attempts = Arc::clone(&builder.attempts);
        let bodies = Arc::clone(&builder.bodies);
        let operator = Operator::new(builder).unwrap();
        let store = store_with_retry(operator);

        let mut upload = store.put_multipart(&Path::from("object")).await.unwrap();
        upload
            .put_part(PutPayload::from_static(b"payload"))
            .await
            .unwrap();
        upload.complete().await.unwrap();

        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert_eq!(
            bodies.lock().unwrap().as_slice(),
            &[
                Bytes::from_static(b"payload"),
                Bytes::from_static(b"payload")
            ]
        );
    }
}
