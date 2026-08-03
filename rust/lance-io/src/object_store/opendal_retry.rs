// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use object_store_opendal::OpendalStore;
use opendal::raw::{
    BoxedFuture, Layer, OpCopier, OpCopy, OpCreateDir, OpList, OpPresign, OpRead, OpRename, OpStat,
    OpWrite, RpCreateDir, RpPresign, RpRename, RpStat, Service, ServiceInfo, Servicer, oio,
};
use opendal::{Buffer, Capability, Metadata, OperationContext, Operator, Result};
use rand::Rng;

fn max_retries() -> usize {
    static MAX_RETRIES: OnceLock<usize> = OnceLock::new();
    *MAX_RETRIES.get_or_init(|| {
        std::env::var("LANCE_CONN_RESET_RETRIES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(20)
    })
}

async fn retry_writer_operation<T, W>(
    writer: &mut W,
    mut operation: impl for<'a> FnMut(&'a mut W) -> BoxedFuture<'a, Result<T>>,
) -> Result<T>
where
    W: oio::Write,
{
    let mut retries = 0;
    loop {
        match operation(writer).await {
            Ok(value) => return Ok(value),
            Err(error) if error.is_temporary() && retries < max_retries() => {
                retries += 1;
                let delay = Duration::from_millis(rand::rng().random_range(2_000..8_000));
                log::warn!(
                    "Retrying OpenDAL writer operation after temporary error (attempt {}): {:?}",
                    retries,
                    error
                );
                tokio::time::sleep(delay).await;
            }
            Err(error) => return Err(error.set_persistent()),
        }
    }
}

#[derive(Debug)]
struct RetryWriter<W> {
    inner: W,
}

impl<W: oio::Write> oio::Write for RetryWriter<W> {
    async fn write(&mut self, body: Buffer) -> Result<()> {
        retry_writer_operation(&mut self.inner, move |writer| {
            Box::pin(writer.write(body.clone()))
        })
        .await
    }

    async fn close(&mut self) -> Result<Metadata> {
        retry_writer_operation(&mut self.inner, |writer| Box::pin(writer.close())).await
    }

    async fn abort(&mut self) -> Result<()> {
        retry_writer_operation(&mut self.inner, |writer| Box::pin(writer.abort())).await
    }
}

#[derive(Clone, Debug)]
struct WriterRetryService {
    inner: Servicer,
}

impl Service for WriterRetryService {
    type Reader = oio::Reader;
    type Writer = RetryWriter<oio::Writer>;
    type Lister = oio::Lister;
    type Deleter = oio::Deleter;
    type Copier = oio::Copier;

    fn info(&self) -> ServiceInfo {
        self.inner.info()
    }

    fn capability(&self) -> Capability {
        self.inner.capability()
    }

    async fn create_dir(
        &self,
        ctx: &OperationContext,
        path: &str,
        args: OpCreateDir,
    ) -> Result<RpCreateDir> {
        self.inner.create_dir(ctx, path, args).await
    }

    async fn stat(&self, ctx: &OperationContext, path: &str, args: OpStat) -> Result<RpStat> {
        self.inner.stat(ctx, path, args).await
    }

    fn read(&self, ctx: &OperationContext, path: &str, args: OpRead) -> Result<Self::Reader> {
        self.inner.read(ctx, path, args)
    }

    fn write(&self, ctx: &OperationContext, path: &str, args: OpWrite) -> Result<Self::Writer> {
        Ok(RetryWriter {
            inner: self.inner.write(ctx, path, args)?,
        })
    }

    fn delete(&self, ctx: &OperationContext) -> Result<Self::Deleter> {
        self.inner.delete(ctx)
    }

    fn list(&self, ctx: &OperationContext, path: &str, args: OpList) -> Result<Self::Lister> {
        self.inner.list(ctx, path, args)
    }

    fn copy(
        &self,
        ctx: &OperationContext,
        from: &str,
        to: &str,
        args: OpCopy,
        opts: OpCopier,
    ) -> Result<Self::Copier> {
        self.inner.copy(ctx, from, to, args, opts)
    }

    async fn rename(
        &self,
        ctx: &OperationContext,
        from: &str,
        to: &str,
        args: OpRename,
    ) -> Result<RpRename> {
        self.inner.rename(ctx, from, to, args).await
    }

    async fn presign(
        &self,
        ctx: &OperationContext,
        path: &str,
        args: OpPresign,
    ) -> Result<RpPresign> {
        self.inner.presign(ctx, path, args).await
    }
}

#[derive(Clone, Copy, Debug)]
struct WriterRetryLayer;

impl Layer for WriterRetryLayer {
    fn apply_service(&self, inner: Servicer) -> Servicer {
        Arc::new(WriterRetryService { inner })
    }
}

/// Install retries beneath OpenDAL's writer so a retry reuses the same writer
/// state and buffer instead of allocating another cloud multipart part. Other
/// operations remain unwrapped so they report throttle feedback to AIMD.
pub fn store_with_retry(operator: Operator) -> OpendalStore {
    OpendalStore::new(operator.layer(WriterRetryLayer))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use bytes::Bytes;
    use object_store::path::Path;
    use object_store::{ObjectStoreExt, PutPayload};
    use opendal::raw::oio;
    use opendal::{
        Buffer, Builder, Capability, EntryMode, Error, ErrorKind, Metadata, OperationContext,
        Result,
    };

    use super::*;

    #[derive(Clone, Debug, Default)]
    struct TransientWriteBuilder {
        write_attempts: Arc<AtomicUsize>,
        writers_created: Arc<AtomicUsize>,
        stat_attempts: Arc<AtomicUsize>,
        bodies: Arc<Mutex<Vec<Bytes>>>,
    }

    impl Builder for TransientWriteBuilder {
        type Config = ();

        fn build(self) -> Result<impl Service> {
            Ok(TransientWriteService {
                write_attempts: self.write_attempts,
                writers_created: self.writers_created,
                stat_attempts: self.stat_attempts,
                bodies: self.bodies,
            })
        }
    }

    #[derive(Clone, Debug)]
    struct TransientWriteService {
        write_attempts: Arc<AtomicUsize>,
        writers_created: Arc<AtomicUsize>,
        stat_attempts: Arc<AtomicUsize>,
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
            self.stat_attempts.fetch_add(1, Ordering::SeqCst);
            Err(Error::new(ErrorKind::Unexpected, "429 Too Many Requests").set_temporary())
        }

        fn read(&self, _: &OperationContext, _: &str, _: OpRead) -> Result<Self::Reader> {
            Err(Error::new(ErrorKind::Unsupported, "read is unsupported"))
        }

        fn write(&self, _: &OperationContext, _: &str, _: OpWrite) -> Result<Self::Writer> {
            self.writers_created.fetch_add(1, Ordering::SeqCst);
            Ok(TransientWriter {
                attempts: Arc::clone(&self.write_attempts),
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
        let write_attempts = Arc::clone(&builder.write_attempts);
        let writers_created = Arc::clone(&builder.writers_created);
        let bodies = Arc::clone(&builder.bodies);
        let operator = Operator::new(builder).unwrap();
        let store = store_with_retry(operator);

        let mut upload = store.put_multipart(&Path::from("object")).await.unwrap();
        upload
            .put_part(PutPayload::from_static(b"payload"))
            .await
            .unwrap();
        upload.complete().await.unwrap();

        assert_eq!(writers_created.load(Ordering::SeqCst), 1);
        assert_eq!(write_attempts.load(Ordering::SeqCst), 2);
        assert_eq!(
            bodies.lock().unwrap().as_slice(),
            &[
                Bytes::from_static(b"payload"),
                Bytes::from_static(b"payload")
            ]
        );
    }

    #[tokio::test]
    async fn test_retry_does_not_wrap_non_writer_operations() {
        let builder = TransientWriteBuilder::default();
        let stat_attempts = Arc::clone(&builder.stat_attempts);
        let operator = Operator::new(builder).unwrap();
        let store = store_with_retry(operator);

        let error = store.head(&Path::from("object")).await.unwrap_err();

        assert!(error.to_string().contains("429 Too Many Requests"));
        assert_eq!(stat_attempts.load(Ordering::SeqCst), 1);
    }
}
