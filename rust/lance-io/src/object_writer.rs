// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::io;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::task::Poll;

use crate::object_store::ObjectStore as LanceObjectStore;
use async_trait::async_trait;
use bytes::Bytes;
use futures::FutureExt;
use futures::future::BoxFuture;
use object_store::{Error as OSError, ObjectStore, Result as OSResult, path::Path};
use object_store::{MultipartUpload, ObjectStoreExt};
use rand::Rng;
use tokio::io::{AsyncWrite, AsyncWriteExt};
use tokio::sync::Mutex;
use tokio::task::JoinSet;

use lance_core::{Error, Result};
use tracing::Instrument;

use crate::traits::Writer;
use crate::utils::tracking_store::IOTracker;
use tokio::runtime::Handle;

/// Start at 5MB.
const INITIAL_UPLOAD_STEP: usize = 1024 * 1024 * 5;

fn max_upload_parallelism() -> usize {
    static MAX_UPLOAD_PARALLELISM: OnceLock<usize> = OnceLock::new();
    *MAX_UPLOAD_PARALLELISM.get_or_init(|| {
        std::env::var("LANCE_UPLOAD_CONCURRENCY")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(10)
    })
}

fn max_conn_reset_retries() -> u16 {
    static MAX_CONN_RESET_RETRIES: OnceLock<u16> = OnceLock::new();
    *MAX_CONN_RESET_RETRIES.get_or_init(|| {
        std::env::var("LANCE_CONN_RESET_RETRIES")
            .ok()
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(20)
    })
}

/// Maximum body size for a single S3 PUT: strictly less than 5 GiB.
/// AWS rejects single-PUT bodies of exactly 5 GiB (= 5 * 1024^3) with
/// `EntityTooLarge`, so we clamp `LANCE_INITIAL_UPLOAD_SIZE` one byte
/// below that threshold to keep the buffer-fills-to-clamp single-PUT
/// path safe. See lance#6750 for the related txn-file write fix.
const MAX_UPLOAD_PART_SIZE: usize = 1024 * 1024 * 1024 * 5 - 1;

/// Clamps a requested upload part size to the valid [5MB, 5GB] range.
/// Returns the clamped value and whether clamping was necessary.
fn clamp_initial_upload_size(raw: usize) -> (usize, bool) {
    let clamped = raw.clamp(INITIAL_UPLOAD_STEP, MAX_UPLOAD_PART_SIZE);
    (clamped, clamped != raw)
}

fn initial_upload_size() -> usize {
    static LANCE_INITIAL_UPLOAD_SIZE: OnceLock<usize> = OnceLock::new();
    *LANCE_INITIAL_UPLOAD_SIZE.get_or_init(|| {
        let Some(raw) = std::env::var("LANCE_INITIAL_UPLOAD_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        else {
            return INITIAL_UPLOAD_STEP;
        };
        let (clamped, was_clamped) = clamp_initial_upload_size(raw);
        if was_clamped {
            // OnceLock caches the result, so this warning fires at most once per process.
            tracing::warn!(
                requested = raw,
                clamped,
                "LANCE_INITIAL_UPLOAD_SIZE must be between 5MB and 5GB; clamping to valid range"
            );
        }
        clamped
    })
}

/// Writer to an object in an object store.
///
/// If the object is small enough, the writer will upload the object in a single
/// PUT request. If the object is larger, the writer will create a multipart
/// upload and upload parts in parallel.
///
/// This implements the `AsyncWrite` trait.
pub struct ObjectWriter {
    state: UploadState,
    // Kept separately from `state` so multipart resources remain owned by the
    // writer if an abort future is cancelled or backend cleanup fails.
    abort_incomplete: bool,
    abort_error: Option<String>,
    path: Arc<Path>,
    cursor: usize,
    connection_resets: u16,
    buffer: Vec<u8>,
    // TODO: use constant size to support R2
    use_constant_size_upload_parts: bool,
}

#[derive(Debug, Clone, Default)]
pub struct WriteResult {
    pub size: usize,
    pub e_tag: Option<String>,
}

enum UploadState {
    /// The writer has been opened but no data has been written yet. Will be in
    /// this state until the buffer is full or the writer is shut down.
    Started(Arc<dyn ObjectStore>),
    /// The writer is in the process of creating a multipart upload.
    CreatingUpload(BoxFuture<'static, OSResult<Box<dyn MultipartUpload>>>),
    /// The writer is in the process of uploading parts.
    InProgress {
        part_idx: u16,
        upload: Box<dyn MultipartUpload>,
        futures: JoinSet<std::result::Result<(), UploadPutError>>,
    },
    /// The writer is in the process of uploading data in a single PUT request.
    /// This happens when shutdown is called before the buffer is full.
    PuttingSingle(BoxFuture<'static, OSResult<WriteResult>>),
    /// The writer is in the process of completing the multipart upload.
    Completing {
        upload: Arc<Mutex<Box<dyn MultipartUpload>>>,
        future: BoxFuture<'static, OSResult<WriteResult>>,
    },
    /// Multipart completion failed and the upload is being aborted before the
    /// completion error is returned to the caller.
    AbortingAfterCompletionFailure {
        upload: Arc<Mutex<Box<dyn MultipartUpload>>>,
        future: BoxFuture<'static, (io::Error, bool)>,
    },
    /// A terminal upload failure. The original error is returned once and its
    /// message is retained so later operations fail instead of polling a
    /// completed future.
    Failed(String),
    /// A terminal part-upload failure that still owns multipart resources.
    /// Later I/O returns `message`; abort or drop cleans up the upload.
    FailedMultipart {
        message: String,
        upload: Box<dyn MultipartUpload>,
        futures: JoinSet<std::result::Result<(), UploadPutError>>,
    },
    /// A terminal completion failure whose failed automatic abort can be retried.
    FailedMultipartCompletion {
        message: String,
        upload: Arc<Mutex<Box<dyn MultipartUpload>>>,
    },
    /// The writer was explicitly aborted and cannot be reused.
    Aborted,
    /// The writer has been shut down and all data has been written.
    Done(WriteResult),
}

/// Methods for state transitions.
impl UploadState {
    fn started_to_putting_single(&mut self, path: Arc<Path>, buffer: Vec<u8>) {
        // To get owned self, we temporarily swap with Done.
        let this = std::mem::replace(self, Self::Done(WriteResult::default()));
        *self = match this {
            Self::Started(store) => {
                let fut = async move {
                    let size = buffer.len();
                    let res = store.put(&path, buffer.into()).await?;
                    Ok(WriteResult {
                        size,
                        e_tag: res.e_tag,
                    })
                };
                Self::PuttingSingle(Box::pin(fut))
            }
            _ => unreachable!(),
        }
    }

    fn in_progress_to_completing(&mut self) {
        // To get owned self, we temporarily swap with Done.
        let this = std::mem::replace(self, Self::Done(WriteResult::default()));
        *self = match this {
            Self::InProgress {
                upload, futures, ..
            } => {
                debug_assert!(
                    futures.is_empty(),
                    "multipart completion requires all part-upload tasks to finish"
                );
                let upload = Arc::new(Mutex::new(upload));
                let completing_upload = upload.clone();
                let fut = async move {
                    let res = completing_upload.lock().await.complete().await?;
                    Ok(WriteResult {
                        size: 0, // This will be set properly later.
                        e_tag: res.e_tag,
                    })
                };
                Self::Completing {
                    upload,
                    future: Box::pin(fut),
                }
            }
            _ => unreachable!(),
        };
    }
}

impl ObjectWriter {
    pub async fn new(object_store: &LanceObjectStore, path: &Path) -> Result<Self> {
        Ok(Self {
            state: UploadState::Started(object_store.inner.clone()),
            abort_incomplete: false,
            abort_error: None,
            cursor: 0,
            path: Arc::new(path.clone()),
            connection_resets: 0,
            buffer: Vec::with_capacity(initial_upload_size()),
            use_constant_size_upload_parts: object_store.use_constant_size_upload_parts,
        })
    }

    /// Returns the contents of `buffer` as a `Bytes` object and resets `buffer`.
    /// The new capacity of `buffer` is determined by the current part index.
    fn next_part_buffer(buffer: &mut Vec<u8>, part_idx: u16, constant_upload_size: bool) -> Bytes {
        let new_capacity = if constant_upload_size {
            // The store does not support variable part sizes, so use the initial size.
            initial_upload_size()
        } else {
            // Increase the upload size every 100 parts. This gives maximum part size of 2.5TB.
            initial_upload_size().max(((part_idx / 100) as usize + 1) * INITIAL_UPLOAD_STEP)
        };
        let new_buffer = Vec::with_capacity(new_capacity);
        let part = std::mem::replace(buffer, new_buffer);
        Bytes::from(part)
    }

    fn put_part(
        upload: &mut dyn MultipartUpload,
        buffer: Bytes,
        part_idx: u16,
        sleep: Option<std::time::Duration>,
    ) -> BoxFuture<'static, std::result::Result<(), UploadPutError>> {
        log::debug!(
            "MultipartUpload submitting part with {} bytes",
            buffer.len()
        );
        let fut = upload.put_part(buffer.clone().into());
        Box::pin(async move {
            if let Some(sleep) = sleep {
                tokio::time::sleep(sleep).await;
            }
            fut.await.map_err(|source| UploadPutError {
                part_idx,
                buffer,
                source,
            })?;
            Ok(())
        })
    }

    fn poll_tasks(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::result::Result<(), io::Error> {
        let mut_self = &mut *self;
        if let Some(message) = &mut_self.abort_error {
            return Err(io::Error::other(message.clone()));
        }
        if mut_self.abort_incomplete {
            return Err(io::Error::other(format!(
                "object writer for '{}' was aborted",
                mut_self.path
            )));
        }
        loop {
            match &mut mut_self.state {
                UploadState::Started(_) | UploadState::Done(_) => break,
                UploadState::Aborted => {
                    return Err(io::Error::other(format!(
                        "object writer for '{}' was aborted",
                        mut_self.path
                    )));
                }
                UploadState::Failed(message)
                | UploadState::FailedMultipart { message, .. }
                | UploadState::FailedMultipartCompletion { message, .. } => {
                    return Err(io::Error::other(message.clone()));
                }
                UploadState::CreatingUpload(fut) => match fut.poll_unpin(cx) {
                    Poll::Ready(Ok(mut upload)) => {
                        let mut futures = JoinSet::new();

                        let data = Self::next_part_buffer(
                            &mut mut_self.buffer,
                            0,
                            mut_self.use_constant_size_upload_parts,
                        );
                        futures.spawn(Self::put_part(upload.as_mut(), data, 0, None));

                        mut_self.state = UploadState::InProgress {
                            part_idx: 1, // We just used 0
                            futures,
                            upload,
                        };
                    }
                    Poll::Ready(Err(e)) => {
                        let error = io::Error::other(e);
                        mut_self.state = UploadState::Failed(error.to_string());
                        return Err(error);
                    }
                    Poll::Pending => break,
                },
                UploadState::InProgress {
                    upload, futures, ..
                } => {
                    let mut upload_error = None;
                    while let Poll::Ready(Some(res)) = futures.poll_join_next(cx) {
                        match res {
                            Ok(Ok(())) => {}
                            Err(err) => {
                                upload_error = Some(io::Error::other(err));
                                break;
                            }
                            Ok(Err(err)) if should_retry_upload_put(&err.source) => {
                                if mut_self.connection_resets < max_conn_reset_retries() {
                                    // Retry, but only up to max_conn_reset_retries of them.
                                    mut_self.connection_resets += 1;

                                    // Resubmit with random jitter
                                    let sleep_time_ms = rand::rng().random_range(2_000..8_000);
                                    let sleep_time =
                                        std::time::Duration::from_millis(sleep_time_ms);

                                    futures.spawn(Self::put_part(
                                        upload.as_mut(),
                                        err.buffer,
                                        err.part_idx,
                                        Some(sleep_time),
                                    ));
                                } else {
                                    upload_error = Some(io::Error::new(
                                        io::ErrorKind::ConnectionReset,
                                        Box::new(ConnectionResetError {
                                            message: format!(
                                                "Hit max retries ({}) for retryable upload error",
                                                max_conn_reset_retries()
                                            ),
                                            source: Box::new(err.source),
                                        }),
                                    ));
                                    break;
                                }
                            }
                            Ok(Err(err)) => {
                                upload_error = Some(err.source.into());
                                break;
                            }
                        }
                    }
                    if let Some(error) = upload_error {
                        let message = error.to_string();
                        let state = std::mem::replace(
                            &mut mut_self.state,
                            UploadState::Failed(message.clone()),
                        );
                        let UploadState::InProgress {
                            upload, futures, ..
                        } = state
                        else {
                            unreachable!()
                        };
                        mut_self.state = UploadState::FailedMultipart {
                            message,
                            upload,
                            futures,
                        };
                        return Err(error);
                    }
                    break;
                }
                UploadState::PuttingSingle(fut) => match fut.poll_unpin(cx) {
                    Poll::Ready(Ok(mut res)) => {
                        res.size = mut_self.cursor;
                        mut_self.state = UploadState::Done(res)
                    }
                    Poll::Ready(Err(e)) => {
                        let error = io::Error::other(e);
                        mut_self.state = UploadState::Failed(error.to_string());
                        return Err(error);
                    }
                    Poll::Pending => break,
                },
                UploadState::Completing { upload, future } => match future.poll_unpin(cx) {
                    Poll::Ready(Ok(mut res)) => {
                        res.size = mut_self.cursor;
                        mut_self.state = UploadState::Done(res)
                    }
                    Poll::Ready(Err(completion_error)) => {
                        let upload = upload.clone();
                        let aborting_upload = upload.clone();
                        let future = async move {
                            let abort_error = aborting_upload.lock().await.abort().await.err();
                            let abort_failed = abort_error.is_some();
                            (
                                io::Error::other(MultipartCompletionError {
                                    completion_error,
                                    abort_error,
                                }),
                                abort_failed,
                            )
                        };
                        mut_self.state = UploadState::AbortingAfterCompletionFailure {
                            upload,
                            future: Box::pin(future),
                        };
                    }
                    Poll::Pending => break,
                },
                UploadState::AbortingAfterCompletionFailure { upload, future } => {
                    match future.poll_unpin(cx) {
                        Poll::Ready((error, abort_failed)) => {
                            if abort_failed {
                                mut_self.state = UploadState::FailedMultipartCompletion {
                                    message: error.to_string(),
                                    upload: upload.clone(),
                                };
                            } else {
                                mut_self.state = UploadState::Failed(error.to_string());
                            }
                            return Err(error);
                        }
                        Poll::Pending => break,
                    }
                }
            }
        }
        Ok(())
    }

    async fn abort_inner(&mut self) -> Result<()> {
        if matches!(
            self.state,
            UploadState::Done(_) | UploadState::Failed(_) | UploadState::Aborted
        ) {
            return Ok(());
        }

        // Reject I/O before the first await, but retain the active state until
        // cleanup succeeds. Cancellation and backend errors therefore leave
        // enough information for a later abort or Drop to retry cleanup.
        self.abort_incomplete = true;
        self.abort_error = None;
        let prior_failure = match &self.state {
            UploadState::FailedMultipart { message, .. }
            | UploadState::FailedMultipartCompletion { message, .. } => Some(message.clone()),
            _ => None,
        };

        let result = match &mut self.state {
            UploadState::InProgress {
                upload, futures, ..
            }
            | UploadState::FailedMultipart {
                upload, futures, ..
            } => abort_in_progress_upload_mut(upload.as_mut(), futures).await,
            UploadState::Completing { upload, future } => {
                // Dropping the completion/abort future releases its lock on the
                // multipart handle before this cleanup attempt acquires it.
                drop(std::mem::replace(
                    future,
                    futures::future::pending::<OSResult<WriteResult>>().boxed(),
                ));
                upload.lock().await.abort().await
            }
            UploadState::AbortingAfterCompletionFailure { upload, future } => {
                drop(std::mem::replace(
                    future,
                    futures::future::pending::<(io::Error, bool)>().boxed(),
                ));
                upload.lock().await.abort().await
            }
            UploadState::FailedMultipartCompletion { upload, .. } => {
                upload.lock().await.abort().await
            }
            UploadState::Started(_)
            | UploadState::CreatingUpload(_)
            | UploadState::PuttingSingle(_) => Ok(()),
            UploadState::Done(_) | UploadState::Failed(_) | UploadState::Aborted => unreachable!(),
        };

        match result {
            Ok(()) => {
                self.abort_incomplete = false;
                self.state = prior_failure
                    .map(UploadState::Failed)
                    .unwrap_or(UploadState::Aborted);
                Ok(())
            }
            Err(error) => {
                let message = if let Some(prior_failure) = prior_failure {
                    format!(
                        "{prior_failure}; additionally failed to abort multipart upload: {error}"
                    )
                } else {
                    error.to_string()
                };
                self.abort_error = Some(message);
                Err(error.into())
            }
        }
    }

    /// Abort an unfinished write.
    ///
    /// This method predates [`Writer::abort`] and therefore cannot return a
    /// backend cleanup failure without changing its public signature. New code
    /// should call [`Writer::abort`] to observe such failures. Ordinary method
    /// syntax (`writer.abort().await`) selects this error-swallowing inherent
    /// method, so use fully qualified syntax when the error matters:
    ///
    /// ```no_run
    /// # use lance_core::Result;
    /// # use lance_io::object_writer::ObjectWriter;
    /// # use lance_io::traits::Writer;
    /// # async fn discard(mut writer: ObjectWriter) -> Result<()> {
    /// Writer::abort(&mut writer).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn abort(&mut self) {
        if let Err(error) = self.abort_inner().await {
            tracing::warn!(path = %self.path, %error, "failed to abort multipart upload");
        }
    }
}

async fn abort_in_progress_upload(
    mut upload: Box<dyn MultipartUpload>,
    mut futures: JoinSet<std::result::Result<(), UploadPutError>>,
) -> OSResult<()> {
    abort_in_progress_upload_mut(upload.as_mut(), &mut futures).await
}

async fn abort_in_progress_upload_mut(
    upload: &mut dyn MultipartUpload,
    futures: &mut JoinSet<std::result::Result<(), UploadPutError>>,
) -> OSResult<()> {
    futures.abort_all();
    while futures.join_next().await.is_some() {}
    upload.abort().await
}

impl Drop for ObjectWriter {
    fn drop(&mut self) {
        let state = std::mem::replace(&mut self.state, UploadState::Done(WriteResult::default()));
        let path = self.path.clone();
        let Ok(handle) = Handle::try_current() else {
            return;
        };

        match state {
            UploadState::InProgress {
                upload, futures, ..
            }
            | UploadState::FailedMultipart {
                upload, futures, ..
            } => {
                handle.spawn(async move {
                    if let Err(error) = abort_in_progress_upload(upload, futures).await {
                        tracing::warn!(path = %path, %error, "failed to abort dropped multipart upload");
                    }
                });
            }
            UploadState::Completing { upload, future } => {
                drop(future);
                handle.spawn(async move {
                    if let Err(error) = upload.lock().await.abort().await {
                        tracing::warn!(path = %path, %error, "failed to abort dropped multipart upload");
                    }
                });
            }
            UploadState::AbortingAfterCompletionFailure { upload, future } => {
                drop(future);
                handle.spawn(async move {
                    if let Err(error) = upload.lock().await.abort().await {
                        tracing::warn!(path = %path, %error, "failed to abort dropped multipart upload");
                    }
                });
            }
            UploadState::FailedMultipartCompletion { upload, .. } => {
                handle.spawn(async move {
                    if let Err(error) = upload.lock().await.abort().await {
                        tracing::warn!(path = %path, %error, "failed to abort dropped multipart upload");
                    }
                });
            }
            UploadState::Started(_)
            | UploadState::CreatingUpload(_)
            | UploadState::PuttingSingle(_)
            | UploadState::Done(_)
            | UploadState::Failed(_)
            | UploadState::Aborted => {}
        }
    }
}

#[derive(Debug)]
struct MultipartCompletionError {
    completion_error: OSError,
    abort_error: Option<OSError>,
}

impl std::fmt::Display for MultipartCompletionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "failed to complete multipart upload: {}",
            self.completion_error
        )?;
        if let Some(abort_error) = &self.abort_error {
            write!(
                f,
                "; additionally failed to abort multipart upload: {}",
                abort_error
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for MultipartCompletionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.completion_error)
    }
}

/// Returned error from trying to upload a part.
/// Has the part_idx and buffer so we can pass
/// them to the retry logic.
struct UploadPutError {
    part_idx: u16,
    buffer: Bytes,
    source: OSError,
}

fn should_retry_upload_put(source: &OSError) -> bool {
    let OSError::Generic { source, .. } = source else {
        return false;
    };

    let message = source.to_string().to_ascii_lowercase();
    message.contains("connection reset by peer") || message.contains("requesttimeout")
}

#[derive(Debug)]
struct ConnectionResetError {
    message: String,
    source: Box<dyn std::error::Error + Send + Sync>,
}

impl std::error::Error for ConnectionResetError {}

impl std::fmt::Display for ConnectionResetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.message, self.source)
    }
}

impl AsyncWrite for ObjectWriter {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::result::Result<usize, std::io::Error>> {
        self.as_mut().poll_tasks(cx)?;

        // Fill buffer up to remaining capacity.
        let remaining_capacity = self.buffer.capacity() - self.buffer.len();
        let bytes_to_write = std::cmp::min(remaining_capacity, buf.len());
        self.buffer.extend_from_slice(&buf[..bytes_to_write]);
        self.cursor += bytes_to_write;

        // Rust needs a little help to borrow self mutably and immutably at the same time
        // through a Pin.
        let mut_self = &mut *self;

        // Instantiate next request, if available.
        if mut_self.buffer.capacity() == mut_self.buffer.len() {
            match &mut mut_self.state {
                UploadState::Started(store) => {
                    let path = mut_self.path.clone();
                    let store = store.clone();
                    let fut = Box::pin(async move { store.put_multipart(path.as_ref()).await });
                    self.state = UploadState::CreatingUpload(fut);
                }
                // TODO: Make max concurrency configurable from storage options.
                UploadState::InProgress {
                    upload,
                    part_idx,
                    futures,
                    ..
                } if futures.len() < max_upload_parallelism() => {
                    let data = Self::next_part_buffer(
                        &mut mut_self.buffer,
                        *part_idx,
                        mut_self.use_constant_size_upload_parts,
                    );
                    futures.spawn(
                        Self::put_part(upload.as_mut(), data, *part_idx, None)
                            .instrument(tracing::Span::current()),
                    );
                    *part_idx += 1;
                }
                _ => {}
            }
        }

        self.poll_tasks(cx)?;

        match bytes_to_write {
            0 => Poll::Pending,
            _ => Poll::Ready(Ok(bytes_to_write)),
        }
    }

    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::result::Result<(), std::io::Error>> {
        self.as_mut().poll_tasks(cx)?;

        match &self.state {
            UploadState::Started(_) | UploadState::Done(_) => Poll::Ready(Ok(())),
            UploadState::CreatingUpload(_)
            | UploadState::Completing { .. }
            | UploadState::AbortingAfterCompletionFailure { .. }
            | UploadState::PuttingSingle(_) => Poll::Pending,
            UploadState::InProgress { futures, .. } => {
                if futures.is_empty() {
                    Poll::Ready(Ok(()))
                } else {
                    Poll::Pending
                }
            }
            UploadState::Failed(message)
            | UploadState::FailedMultipart { message, .. }
            | UploadState::FailedMultipartCompletion { message, .. } => {
                Poll::Ready(Err(io::Error::other(message.clone())))
            }
            UploadState::Aborted => Poll::Ready(Err(io::Error::other(format!(
                "object writer for '{}' was aborted",
                self.path
            )))),
        }
    }

    fn poll_shutdown(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::result::Result<(), std::io::Error>> {
        loop {
            self.as_mut().poll_tasks(cx)?;

            // Rust needs a little help to borrow self mutably and immutably at the same time
            // through a Pin.
            let mut_self = &mut *self;
            match &mut mut_self.state {
                UploadState::Done(_) => return Poll::Ready(Ok(())),
                UploadState::CreatingUpload(_)
                | UploadState::PuttingSingle(_)
                | UploadState::Completing { .. }
                | UploadState::AbortingAfterCompletionFailure { .. } => return Poll::Pending,
                UploadState::Failed(message)
                | UploadState::FailedMultipart { message, .. }
                | UploadState::FailedMultipartCompletion { message, .. } => {
                    return Poll::Ready(Err(io::Error::other(message.clone())));
                }
                UploadState::Aborted => {
                    return Poll::Ready(Err(io::Error::other(format!(
                        "object writer for '{}' was aborted",
                        mut_self.path
                    ))));
                }
                UploadState::Started(_) => {
                    // If we didn't start a multipart upload, we can just do a single put.
                    let part = std::mem::take(&mut mut_self.buffer);
                    let path = mut_self.path.clone();
                    self.state.started_to_putting_single(path, part);
                }
                UploadState::InProgress {
                    upload,
                    futures,
                    part_idx,
                } => {
                    // Flush final batch
                    if !mut_self.buffer.is_empty() && futures.len() < max_upload_parallelism() {
                        // We can just use `take` since we don't need the buffer anymore.
                        let data = Bytes::from(std::mem::take(&mut mut_self.buffer));
                        futures.spawn(
                            Self::put_part(upload.as_mut(), data, *part_idx, None)
                                .instrument(tracing::Span::current()),
                        );
                        // We need to go back to beginning of loop to poll the
                        // new feature and get the waker registered on the ctx.
                        continue;
                    }

                    // We handle the transition from in progress to completing here.
                    if futures.is_empty() {
                        self.state.in_progress_to_completing();
                    } else {
                        return Poll::Pending;
                    }
                }
            }
        }
    }
}

#[async_trait]
impl Writer for ObjectWriter {
    async fn tell(&mut self) -> Result<usize> {
        Ok(self.cursor)
    }

    async fn shutdown(&mut self) -> Result<WriteResult> {
        AsyncWriteExt::shutdown(self).await.map_err(|e| {
            Error::io(format!(
                "failed to shutdown object writer for {}: {}",
                self.path, e
            ))
        })?;
        if let UploadState::Done(result) = &self.state {
            Ok(result.clone())
        } else {
            unreachable!()
        }
    }

    async fn abort(&mut self) -> Result<()> {
        self.abort_inner().await
    }
}

pub struct LocalWriter {
    path: Path,
    state: LocalWriteState,
    #[cfg(test)]
    persist_hooks: LocalPersistTestHooks,
}

#[derive(Default)]
enum LocalWriteState {
    Writing(WritingState),
    Finishing {
        size: usize,
        // Keep ownership of the temp path in the state machine. The blocking
        // task only prepares metadata, so aborting or dropping this state can
        // delete the temp file without leaving a detached publisher behind.
        temp_path: tempfile::TempPath,
        io_tracker: Arc<IOTracker>,
        future: BoxFuture<'static, Result<String>>,
    },
    Done(WriteResult),
    Failed(String),
    Aborted,
    #[default]
    Poisoned,
}

#[cfg(test)]
#[derive(Default)]
struct LocalPersistTestHooks {
    before_prepare: Option<Box<dyn FnOnce() + Send>>,
    after_prepare: Option<Box<dyn FnOnce() + Send>>,
}

struct WritingState {
    writer: tokio::io::BufWriter<tokio::fs::File>,
    cursor: usize,
    /// Temp path that auto-deletes on drop until it is successfully persisted.
    temp_path: tempfile::TempPath,
    io_tracker: Arc<IOTracker>,
}

impl LocalWriter {
    pub fn new(
        file: tokio::fs::File,
        path: Path,
        temp_path: tempfile::TempPath,
        io_tracker: Arc<IOTracker>,
    ) -> Self {
        Self {
            path,
            state: LocalWriteState::Writing(WritingState {
                writer: tokio::io::BufWriter::new(file),
                cursor: 0,
                temp_path,
                io_tracker,
            }),
            #[cfg(test)]
            persist_hooks: LocalPersistTestHooks::default(),
        }
    }

    fn already_closed_err(path: &Path) -> io::Error {
        io::Error::other(format!(
            "cannot write to LocalWriter for {} after shutdown",
            path
        ))
    }

    fn poisoned_err(path: &Path) -> io::Error {
        io::Error::other(format!("LocalWriter for {} is in poisoned state", path))
    }

    fn aborted_err(path: &Path) -> io::Error {
        io::Error::other(format!("LocalWriter for {} was aborted", path))
    }

    fn failed_err(path: &Path, message: &str) -> io::Error {
        io::Error::other(format!("LocalWriter for {} failed: {}", path, message))
    }

    async fn prepare_persist(
        temp_file_path: std::path::PathBuf,
        final_path: Path,
        #[cfg(test)] mut test_hooks: LocalPersistTestHooks,
    ) -> Result<String> {
        tokio::task::spawn_blocking(move || -> Result<String> {
            #[cfg(test)]
            if let Some(hook) = test_hooks.before_prepare.take() {
                hook();
            }

            let result = std::fs::metadata(&temp_file_path)
                .map(|metadata| get_etag(&metadata))
                .map_err(|error| {
                    Error::io(format!(
                        "failed to prepare temp file {} for {}: {}",
                        temp_file_path.display(),
                        final_path,
                        error
                    ))
                });

            #[cfg(test)]
            if let Some(hook) = test_hooks.after_prepare.take() {
                hook();
            }

            result
        })
        .await
        .map_err(|error| Error::io(format!("spawn_blocking failed: {}", error)))?
    }
}

impl AsyncWrite for LocalWriter {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> Poll<std::result::Result<usize, std::io::Error>> {
        let path = self.path.clone();
        match &mut self.state {
            LocalWriteState::Writing(state) => {
                let poll = Pin::new(&mut state.writer).poll_write(cx, buf);
                if let Poll::Ready(Ok(n)) = &poll {
                    state.cursor += *n;
                }
                poll
            }
            LocalWriteState::Failed(message) => Poll::Ready(Err(Self::failed_err(&path, message))),
            LocalWriteState::Aborted => Poll::Ready(Err(Self::aborted_err(&path))),
            LocalWriteState::Poisoned => Poll::Ready(Err(Self::poisoned_err(&path))),
            LocalWriteState::Finishing { .. } | LocalWriteState::Done(_) => {
                Poll::Ready(Err(Self::already_closed_err(&path)))
            }
        }
    }

    fn poll_flush(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<std::result::Result<(), std::io::Error>> {
        let path = self.path.clone();
        match &mut self.state {
            LocalWriteState::Writing(state) => Pin::new(&mut state.writer).poll_flush(cx),
            LocalWriteState::Failed(message) => Poll::Ready(Err(Self::failed_err(&path, message))),
            LocalWriteState::Aborted => Poll::Ready(Err(Self::aborted_err(&path))),
            LocalWriteState::Poisoned => Poll::Ready(Err(Self::poisoned_err(&path))),
            LocalWriteState::Finishing { .. } | LocalWriteState::Done(_) => {
                Poll::Ready(Err(Self::already_closed_err(&path)))
            }
        }
    }

    fn poll_shutdown(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<std::result::Result<(), std::io::Error>> {
        let mut_self = &mut *self;
        loop {
            match &mut mut_self.state {
                LocalWriteState::Writing(state) => {
                    match Pin::new(&mut state.writer).poll_shutdown(cx) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(Ok(())) => {}
                        Poll::Ready(Err(error)) => {
                            let message = error.to_string();
                            mut_self.state = LocalWriteState::Failed(message);
                            return Poll::Ready(Err(error));
                        }
                    }

                    // Flush is complete. Prepare the ETag off-thread, but retain
                    // the temp path here so cancellation cannot detach a rename.
                    #[cfg(test)]
                    let test_hooks = std::mem::take(&mut mut_self.persist_hooks);
                    let LocalWriteState::Writing(state) =
                        std::mem::replace(&mut mut_self.state, LocalWriteState::Poisoned)
                    else {
                        unreachable!()
                    };
                    let size = state.cursor;
                    let temp_file_path = state.temp_path.to_path_buf();
                    #[cfg(test)]
                    let future = Box::pin(Self::prepare_persist(
                        temp_file_path,
                        mut_self.path.clone(),
                        test_hooks,
                    ));
                    #[cfg(not(test))]
                    let future =
                        Box::pin(Self::prepare_persist(temp_file_path, mut_self.path.clone()));
                    mut_self.state = LocalWriteState::Finishing {
                        size,
                        temp_path: state.temp_path,
                        io_tracker: state.io_tracker,
                        future,
                    };
                }
                LocalWriteState::Finishing { future, .. } => match future.poll_unpin(cx) {
                    Poll::Ready(Ok(e_tag)) => {
                        let LocalWriteState::Finishing {
                            size,
                            temp_path,
                            io_tracker,
                            ..
                        } = std::mem::replace(&mut mut_self.state, LocalWriteState::Poisoned)
                        else {
                            unreachable!()
                        };

                        // This single rename is the commit point. It runs after
                        // metadata preparation and cannot be cancelled between
                        // successful publication and installing the Done state.
                        let local_path = crate::local::to_local_path(&mut_self.path);
                        if let Err(error) = temp_path.persist(&local_path) {
                            let message = format!(
                                "failed to persist temp file to {}: {}",
                                local_path, error.error
                            );
                            mut_self.state = LocalWriteState::Failed(message.clone());
                            return Poll::Ready(Err(io::Error::other(message)));
                        }

                        mut_self.state = LocalWriteState::Done(WriteResult {
                            size,
                            e_tag: Some(e_tag),
                        });
                        io_tracker.record_write("put", mut_self.path.clone(), size as u64);
                    }
                    Poll::Ready(Err(error)) => {
                        let message = error.to_string();
                        mut_self.state = LocalWriteState::Failed(message);
                        return Poll::Ready(Err(io::Error::other(error)));
                    }
                    Poll::Pending => return Poll::Pending,
                },
                LocalWriteState::Done(_) => return Poll::Ready(Ok(())),
                LocalWriteState::Failed(message) => {
                    return Poll::Ready(Err(Self::failed_err(&mut_self.path, message)));
                }
                LocalWriteState::Aborted => {
                    return Poll::Ready(Err(Self::aborted_err(&mut_self.path)));
                }
                LocalWriteState::Poisoned => {
                    return Poll::Ready(Err(Self::poisoned_err(&self.path)));
                }
            }
        }
    }
}

#[async_trait]
impl Writer for LocalWriter {
    async fn tell(&mut self) -> Result<usize> {
        match &mut self.state {
            LocalWriteState::Writing(state) => Ok(state.cursor),
            LocalWriteState::Finishing { size, .. } => Ok(*size),
            LocalWriteState::Done(result) => Ok(result.size),
            LocalWriteState::Failed(message) => Err(Self::failed_err(&self.path, message).into()),
            LocalWriteState::Aborted => Err(Self::aborted_err(&self.path).into()),
            LocalWriteState::Poisoned => Err(Self::poisoned_err(&self.path).into()),
        }
    }

    async fn shutdown(&mut self) -> Result<WriteResult> {
        AsyncWriteExt::shutdown(self).await.map_err(|e| {
            Error::io(format!(
                "failed to shutdown local writer for {}: {}",
                self.path, e
            ))
        })?;

        match &self.state {
            LocalWriteState::Done(result) => Ok(result.clone()),
            _ => unreachable!(),
        }
    }

    async fn abort(&mut self) -> Result<()> {
        match &self.state {
            LocalWriteState::Done(_) => Err(Error::io(format!(
                "cannot abort LocalWriter for {} because the file is already committed",
                self.path
            ))),
            LocalWriteState::Aborted => Ok(()),
            LocalWriteState::Poisoned => Err(Self::poisoned_err(&self.path).into()),
            LocalWriteState::Writing(_)
            | LocalWriteState::Finishing { .. }
            | LocalWriteState::Failed(_) => {
                self.state = LocalWriteState::Aborted;
                Ok(())
            }
        }
    }
}

// Based on object store's implementation.
pub fn get_etag(metadata: &std::fs::Metadata) -> String {
    let inode = get_inode(metadata);
    let size = metadata.len();
    let mtime = metadata
        .modified()
        .ok()
        .and_then(|mtime| mtime.duration_since(std::time::SystemTime::UNIX_EPOCH).ok())
        .unwrap_or_default()
        .as_micros();

    // Use an ETag scheme based on that used by many popular HTTP servers
    // <https://httpd.apache.org/docs/2.2/mod/core.html#fileetag>
    format!("{inode:x}-{mtime:x}-{size:x}")
}

#[cfg(unix)]
fn get_inode(metadata: &std::fs::Metadata) -> u64 {
    std::os::unix::fs::MetadataExt::ino(metadata)
}

#[cfg(not(unix))]
fn get_inode(_metadata: &std::fs::Metadata) -> u64 {
    0
}

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::sync::Mutex as StdMutex;

    use futures::future;
    use object_store::{PutPayload, PutResult, UploadPart};
    use rstest::rstest;
    use tokio::io::AsyncWriteExt;
    use url::Url;

    use super::*;
    use crate::testing::MockObjectStore;

    #[derive(Debug, Clone, Copy)]
    enum PartBehavior {
        Ready,
        Pending,
        Failure,
    }

    #[derive(Debug)]
    struct TestMultipartUpload {
        events: Arc<StdMutex<Vec<&'static str>>>,
        part_behavior: PartBehavior,
        fail_completion: bool,
        abort_failures_remaining: usize,
        pending_abort_once: bool,
    }

    impl TestMultipartUpload {
        fn new(events: Arc<StdMutex<Vec<&'static str>>>, part_behavior: PartBehavior) -> Self {
            Self {
                events,
                part_behavior,
                fail_completion: false,
                abort_failures_remaining: 0,
                pending_abort_once: false,
            }
        }

        fn with_completion_failure(mut self) -> Self {
            self.fail_completion = true;
            self
        }

        fn with_abort_failure(mut self) -> Self {
            self.abort_failures_remaining = 1;
            self
        }

        fn with_pending_abort(mut self) -> Self {
            self.pending_abort_once = true;
            self
        }
    }

    #[async_trait]
    impl MultipartUpload for TestMultipartUpload {
        fn put_part(&mut self, _data: PutPayload) -> UploadPart {
            match self.part_behavior {
                PartBehavior::Ready => future::ready(Ok(())).boxed(),
                PartBehavior::Pending => Box::pin(PendingPart {
                    events: self.events.clone(),
                }),
                PartBehavior::Failure => {
                    future::ready(Err(test_object_store_error("part upload failed"))).boxed()
                }
            }
        }

        async fn complete(&mut self) -> OSResult<PutResult> {
            self.events.lock().unwrap().push("complete");
            if self.fail_completion {
                Err(test_object_store_error("completion failed"))
            } else {
                Ok(PutResult {
                    e_tag: None,
                    version: None,
                })
            }
        }

        async fn abort(&mut self) -> OSResult<()> {
            self.events.lock().unwrap().push("abort");
            if self.pending_abort_once {
                self.pending_abort_once = false;
                return future::pending::<OSResult<()>>().await;
            }
            if self.abort_failures_remaining > 0 {
                self.abort_failures_remaining -= 1;
                Err(test_object_store_error("abort failed"))
            } else {
                Ok(())
            }
        }
    }

    struct PendingPart {
        events: Arc<StdMutex<Vec<&'static str>>>,
    }

    #[derive(Debug, Clone, Copy)]
    enum LocalFinishCancellation {
        Abort,
        Drop,
    }

    impl Future for PendingPart {
        type Output = OSResult<()>;

        fn poll(self: Pin<&mut Self>, _cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
            Poll::Pending
        }
    }

    impl Drop for PendingPart {
        fn drop(&mut self) {
            self.events.lock().unwrap().push("part_cancelled");
        }
    }

    fn test_object_store_error(message: &'static str) -> OSError {
        OSError::Generic {
            store: "test",
            source: Box::new(io::Error::other(message)),
        }
    }

    fn test_lance_object_store(store: MockObjectStore) -> LanceObjectStore {
        LanceObjectStore::new(
            Arc::new(store),
            Url::parse("memory://").unwrap(),
            None,
            None,
            false,
            true,
            1,
            0,
            None,
        )
    }

    async fn writer_with_multipart_upload(upload: TestMultipartUpload) -> ObjectWriter {
        let mut store = MockObjectStore::new();
        store
            .expect_put_multipart_opts()
            .times(1)
            .return_once(move |_, _| Ok(Box::new(upload)));
        let store = test_lance_object_store(store);
        let mut writer = ObjectWriter::new(&store, &Path::from("/multipart"))
            .await
            .unwrap();
        writer.buffer = Vec::with_capacity(INITIAL_UPLOAD_STEP);
        writer
            .write_all(&vec![0; INITIAL_UPLOAD_STEP])
            .await
            .unwrap();
        assert!(matches!(&writer.state, UploadState::InProgress { .. }));
        writer
    }

    #[tokio::test]
    async fn test_write() {
        let store = LanceObjectStore::memory();

        let mut object_writer = ObjectWriter::new(&store, &Path::from("/foo"))
            .await
            .unwrap();
        assert_eq!(object_writer.tell().await.unwrap(), 0);

        let buf = vec![0; 256];
        assert_eq!(object_writer.write(buf.as_slice()).await.unwrap(), 256);
        assert_eq!(object_writer.tell().await.unwrap(), 256);

        assert_eq!(object_writer.write(buf.as_slice()).await.unwrap(), 256);
        assert_eq!(object_writer.tell().await.unwrap(), 512);

        assert_eq!(object_writer.write(buf.as_slice()).await.unwrap(), 256);
        assert_eq!(object_writer.tell().await.unwrap(), 256 * 3);

        let res = Writer::shutdown(&mut object_writer).await.unwrap();
        assert_eq!(res.size, 256 * 3);

        // Trigger multi part upload
        let mut object_writer = ObjectWriter::new(&store, &Path::from("/bar"))
            .await
            .unwrap();
        let buf = vec![0; INITIAL_UPLOAD_STEP / 3 * 2];
        for i in 0..5 {
            // Write more data to trigger the multipart upload
            // This should be enough to trigger a multipart upload
            object_writer.write_all(buf.as_slice()).await.unwrap();
            // Check the cursor
            assert_eq!(object_writer.tell().await.unwrap(), (i + 1) * buf.len());
        }
        let res = Writer::shutdown(&mut object_writer).await.unwrap();
        assert_eq!(res.size, buf.len() * 5);
    }

    #[tokio::test]
    async fn test_abort_write() {
        let store = LanceObjectStore::memory();

        let mut object_writer = ObjectWriter::new(&store, &Path::from("/foo"))
            .await
            .unwrap();
        Writer::abort(&mut object_writer).await.unwrap();

        let write_error = object_writer.write_all(b"discarded").await.unwrap_err();
        assert!(write_error.to_string().contains("was aborted"));

        let shutdown_error = Writer::shutdown(&mut object_writer).await.unwrap_err();
        assert!(shutdown_error.to_string().contains("was aborted"));

        // Repeated abort remains safe without making the writer usable again.
        Writer::abort(&mut object_writer).await.unwrap();
        assert!(object_writer.write_all(b"still discarded").await.is_err());
    }

    #[tokio::test]
    async fn test_single_put_failure_is_terminal() {
        let mut store = MockObjectStore::new();
        store
            .expect_put_opts()
            .times(1)
            .return_once(|_, _, _| Err(test_object_store_error("single put failed")));
        let store = test_lance_object_store(store);
        let mut writer = ObjectWriter::new(&store, &Path::from("/single"))
            .await
            .unwrap();
        writer.write_all(b"payload").await.unwrap();

        let first_error = Writer::shutdown(&mut writer).await.unwrap_err();
        assert!(first_error.to_string().contains("single put failed"));

        let second_error = Writer::shutdown(&mut writer).await.unwrap_err();
        assert!(second_error.to_string().contains("single put failed"));
        let write_error = writer.write_all(b"more").await.unwrap_err();
        assert!(write_error.to_string().contains("single put failed"));
    }

    #[tokio::test]
    async fn test_multipart_creation_failure_is_terminal() {
        let mut store = MockObjectStore::new();
        store
            .expect_put_multipart_opts()
            .times(1)
            .return_once(|_, _| Err(test_object_store_error("multipart creation failed")));
        let store = test_lance_object_store(store);
        let mut writer = ObjectWriter::new(&store, &Path::from("/multipart"))
            .await
            .unwrap();
        writer.buffer = Vec::with_capacity(INITIAL_UPLOAD_STEP);

        let first_error = writer
            .write_all(&vec![0; INITIAL_UPLOAD_STEP])
            .await
            .unwrap_err();
        assert!(
            first_error
                .to_string()
                .contains("multipart creation failed")
        );

        let write_error = writer.write_all(b"more").await.unwrap_err();
        assert!(
            write_error
                .to_string()
                .contains("multipart creation failed")
        );
        let shutdown_error = Writer::shutdown(&mut writer).await.unwrap_err();
        assert!(
            shutdown_error
                .to_string()
                .contains("multipart creation failed")
        );
    }

    #[tokio::test]
    async fn test_part_failure_is_terminal_and_abortable() {
        let events = Arc::new(StdMutex::new(Vec::new()));
        let upload = TestMultipartUpload::new(events.clone(), PartBehavior::Failure);
        let mut writer = writer_with_multipart_upload(upload).await;

        let first_error = writer.flush().await.unwrap_err();
        assert!(first_error.to_string().contains("part upload failed"));
        let write_error = writer.write_all(b"more").await.unwrap_err();
        assert!(write_error.to_string().contains("part upload failed"));

        Writer::abort(&mut writer).await.unwrap();
        assert_eq!(*events.lock().unwrap(), ["abort"]);
        let shutdown_error = Writer::shutdown(&mut writer).await.unwrap_err();
        assert!(shutdown_error.to_string().contains("part upload failed"));
    }

    #[tokio::test]
    async fn test_completion_failure_aborts_multipart_upload() {
        let events = Arc::new(StdMutex::new(Vec::new()));
        let upload =
            TestMultipartUpload::new(events.clone(), PartBehavior::Ready).with_completion_failure();
        let mut writer = writer_with_multipart_upload(upload).await;

        let error = Writer::shutdown(&mut writer).await.unwrap_err();

        assert!(matches!(&error, Error::IO { .. }));
        assert!(error.to_string().contains("completion failed"));
        assert_eq!(*events.lock().unwrap(), ["complete", "abort"]);
    }

    #[tokio::test]
    async fn test_completion_and_abort_failure_retains_cleanup() {
        let events = Arc::new(StdMutex::new(Vec::new()));
        let upload = TestMultipartUpload::new(events.clone(), PartBehavior::Ready)
            .with_completion_failure()
            .with_abort_failure();
        let mut writer = writer_with_multipart_upload(upload).await;

        let error = Writer::shutdown(&mut writer).await.unwrap_err();
        assert!(error.to_string().contains("completion failed"));
        assert!(error.to_string().contains("abort failed"));
        assert_eq!(*events.lock().unwrap(), ["complete", "abort"]);

        Writer::abort(&mut writer).await.unwrap();
        assert_eq!(*events.lock().unwrap(), ["complete", "abort", "abort"]);
        let shutdown_error = Writer::shutdown(&mut writer).await.unwrap_err();
        assert!(shutdown_error.to_string().contains("completion failed"));
        assert!(shutdown_error.to_string().contains("abort failed"));
    }

    #[tokio::test]
    async fn test_abort_cancels_part_tasks_before_backend_abort() {
        let events = Arc::new(StdMutex::new(Vec::new()));
        let upload = TestMultipartUpload::new(events.clone(), PartBehavior::Pending);
        let mut writer = writer_with_multipart_upload(upload).await;

        Writer::abort(&mut writer).await.unwrap();

        assert_eq!(*events.lock().unwrap(), ["part_cancelled", "abort"]);
    }

    #[tokio::test]
    async fn test_cancelled_abort_remains_terminal_and_can_resume_cleanup() {
        let events = Arc::new(StdMutex::new(Vec::new()));
        let upload =
            TestMultipartUpload::new(events.clone(), PartBehavior::Ready).with_pending_abort();
        let mut writer = writer_with_multipart_upload(upload).await;

        let mut abort = Box::pin(Writer::abort(&mut writer));
        let mut reached_backend_abort = false;
        for _ in 0..100 {
            let poll = future::poll_fn(|cx| Poll::Ready(abort.as_mut().poll(cx))).await;
            assert!(poll.is_pending());
            if *events.lock().unwrap() == ["abort"] {
                reached_backend_abort = true;
                break;
            }
            tokio::task::yield_now().await;
        }
        drop(abort);
        assert!(
            reached_backend_abort,
            "abort did not reach backend cleanup after 100 polls"
        );

        let write_error = writer.write_all(b"discarded").await.unwrap_err();
        assert!(write_error.to_string().contains("was aborted"));

        Writer::abort(&mut writer).await.unwrap();
        assert_eq!(*events.lock().unwrap(), ["abort", "abort"]);
        let shutdown_error = Writer::shutdown(&mut writer).await.unwrap_err();
        assert!(shutdown_error.to_string().contains("was aborted"));
    }

    #[tokio::test]
    async fn test_abort_propagates_backend_error() {
        let events = Arc::new(StdMutex::new(Vec::new()));
        let upload =
            TestMultipartUpload::new(events.clone(), PartBehavior::Ready).with_abort_failure();
        let mut writer = writer_with_multipart_upload(upload).await;

        let error = Writer::abort(&mut writer).await.unwrap_err();

        assert!(matches!(&error, Error::IO { .. }));
        assert!(error.to_string().contains("abort failed"));
        assert_eq!(*events.lock().unwrap(), ["abort"]);

        let write_error = writer.write_all(b"discarded").await.unwrap_err();
        assert!(write_error.to_string().contains("abort failed"));
        let shutdown_error = Writer::shutdown(&mut writer).await.unwrap_err();
        assert!(shutdown_error.to_string().contains("abort failed"));

        // The multipart handle is retained, so cleanup can be retried without
        // ever making the writer writable again.
        Writer::abort(&mut writer).await.unwrap();
        assert_eq!(*events.lock().unwrap(), ["abort", "abort"]);
        let write_error = writer.write_all(b"still discarded").await.unwrap_err();
        assert!(write_error.to_string().contains("was aborted"));
    }

    #[tokio::test]
    async fn test_local_writer_shutdown() {
        let tmp = lance_core::utils::tempfile::TempStdDir::default();
        let file_path = tmp.join("test_local_writer.bin");
        let os_path = Path::from_absolute_path(&file_path).unwrap();
        let io_tracker = Arc::new(IOTracker::default());

        let named_temp = tempfile::NamedTempFile::new_in(&*tmp).unwrap();
        let temp_file_path = named_temp.path().to_owned();
        let (std_file, temp_path) = named_temp.into_parts();
        let file = tokio::fs::File::from_std(std_file);
        let mut writer = LocalWriter::new(file, os_path, temp_path, io_tracker.clone());

        let data = b"hello local writer";
        writer.write_all(data).await.unwrap();

        // Before shutdown, the final path should not exist
        assert!(!file_path.exists());
        // But the temp file should exist
        assert!(temp_file_path.exists());

        let result = Writer::shutdown(&mut writer).await.unwrap();
        assert_eq!(result.size, data.len());
        assert!(result.e_tag.is_some());
        assert!(!result.e_tag.as_ref().unwrap().is_empty());

        // After shutdown, the final path should exist and temp should be gone
        assert!(file_path.exists());
        assert!(!temp_file_path.exists());

        let abort_error = Writer::abort(&mut writer).await.unwrap_err();
        assert!(matches!(&abort_error, Error::IO { .. }));
        assert!(abort_error.to_string().contains("already committed"));
        assert!(file_path.exists());

        let stats = io_tracker.stats();
        assert_eq!(stats.write_iops, 1);
        assert_eq!(stats.written_bytes, data.len() as u64);
    }

    #[tokio::test]
    async fn test_local_writer_abort_removes_temp_file() {
        let tmp = lance_core::utils::tempfile::TempStdDir::default();
        let file_path = tmp.join("aborted.bin");
        let os_path = Path::from_absolute_path(&file_path).unwrap();
        let named_temp = tempfile::NamedTempFile::new_in(&*tmp).unwrap();
        let temp_file_path = named_temp.path().to_owned();
        let (std_file, temp_path) = named_temp.into_parts();
        let file = tokio::fs::File::from_std(std_file);
        let mut writer = LocalWriter::new(file, os_path, temp_path, Arc::new(IOTracker::default()));

        writer.write_all(b"partial").await.unwrap();
        Writer::abort(&mut writer).await.unwrap();

        assert!(!temp_file_path.exists());
        assert!(!file_path.exists());

        let write_error = writer.write_all(b"more").await.unwrap_err();
        assert!(write_error.to_string().contains("was aborted"));
        Writer::abort(&mut writer).await.unwrap();
    }

    #[rstest]
    #[case::abort(LocalFinishCancellation::Abort)]
    #[case::drop(LocalFinishCancellation::Drop)]
    #[tokio::test]
    async fn test_local_writer_cancelled_finishing_cannot_publish(
        #[case] cancellation: LocalFinishCancellation,
    ) {
        let tmp = lance_core::utils::tempfile::TempStdDir::default();
        let file_path = tmp.join("cancelled_finishing.bin");
        let os_path = Path::from_absolute_path(&file_path).unwrap();
        let io_tracker = Arc::new(IOTracker::default());

        let named_temp = tempfile::NamedTempFile::new_in(&*tmp).unwrap();
        let temp_file_path = named_temp.path().to_owned();
        let (std_file, temp_path) = named_temp.into_parts();
        let file = tokio::fs::File::from_std(std_file);
        let mut writer = LocalWriter::new(file, os_path.clone(), temp_path, io_tracker.clone());

        let (prepare_started_tx, prepare_started_rx) = tokio::sync::oneshot::channel();
        let (release_prepare_tx, release_prepare_rx) = std::sync::mpsc::channel();
        let (prepare_finished_tx, prepare_finished_rx) = tokio::sync::oneshot::channel();
        writer.persist_hooks.before_prepare = Some(Box::new(move || {
            prepare_started_tx
                .send(())
                .expect("shutdown future should wait for metadata preparation");
            release_prepare_rx
                .recv()
                .expect("test should release metadata preparation");
        }));
        writer.persist_hooks.after_prepare = Some(Box::new(move || {
            prepare_finished_tx
                .send(())
                .expect("test should wait for detached metadata preparation");
        }));

        writer.write_all(b"stale payload").await.unwrap();
        let mut shutdown = Box::pin(Writer::shutdown(&mut writer));
        tokio::select! {
            result = &mut shutdown => {
                panic!("shutdown completed before the preparation gate: {result:?}");
            }
            result = prepare_started_rx => {
                result.expect("metadata preparation task should start");
            }
            _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => {
                panic!("metadata preparation task did not start");
            }
        }
        drop(shutdown);
        assert!(matches!(&writer.state, LocalWriteState::Finishing { .. }));
        assert!(temp_file_path.exists());

        if matches!(cancellation, LocalFinishCancellation::Abort) {
            Writer::abort(&mut writer).await.unwrap();
            assert!(matches!(&writer.state, LocalWriteState::Aborted));
        }
        drop(writer);

        assert!(!temp_file_path.exists());
        assert!(!file_path.exists());

        // Commit a retry while the cancelled writer's blocking task is still
        // paused. Releasing that old task must not let it overwrite the retry.
        let retry_temp = tempfile::NamedTempFile::new_in(&*tmp).unwrap();
        let (retry_file, retry_path) = retry_temp.into_parts();
        let retry_file = tokio::fs::File::from_std(retry_file);
        let mut retry = LocalWriter::new(
            retry_file,
            os_path,
            retry_path,
            Arc::new(IOTracker::default()),
        );
        retry.write_all(b"fresh payload").await.unwrap();
        Writer::shutdown(&mut retry).await.unwrap();
        assert_eq!(std::fs::read(&file_path).unwrap(), b"fresh payload");

        release_prepare_tx
            .send(())
            .expect("metadata preparation task should still be running");
        tokio::time::timeout(std::time::Duration::from_secs(5), prepare_finished_rx)
            .await
            .expect("detached metadata preparation should finish")
            .expect("detached metadata preparation task should not be cancelled");

        assert_eq!(std::fs::read(&file_path).unwrap(), b"fresh payload");
        assert_eq!(io_tracker.stats().write_iops, 0);
    }

    #[tokio::test]
    async fn test_local_writer_prepare_failure_is_terminal() {
        let tmp = lance_core::utils::tempfile::TempStdDir::default();
        let file_path = tmp.join("prepare_failure.bin");
        let os_path = Path::from_absolute_path(&file_path).unwrap();
        let named_temp = tempfile::NamedTempFile::new_in(&*tmp).unwrap();
        let temp_file_path = named_temp.path().to_owned();
        let (std_file, temp_path) = named_temp.into_parts();
        let file = tokio::fs::File::from_std(std_file);
        let mut writer = LocalWriter::new(file, os_path, temp_path, Arc::new(IOTracker::default()));

        let removed_temp_path = temp_file_path.clone();
        writer.persist_hooks.before_prepare = Some(Box::new(move || {
            std::fs::remove_file(&removed_temp_path)
                .expect("the closed temporary file should be removable");
        }));
        writer.write_all(b"payload").await.unwrap();

        let first_error = Writer::shutdown(&mut writer).await.unwrap_err();
        assert!(matches!(&first_error, Error::IO { .. }));
        assert!(
            first_error
                .to_string()
                .contains("failed to prepare temp file")
        );
        assert!(matches!(&writer.state, LocalWriteState::Failed(_)));

        let second_error = Writer::shutdown(&mut writer).await.unwrap_err();
        assert!(matches!(&second_error, Error::IO { .. }));
        assert!(
            second_error
                .to_string()
                .contains("failed to prepare temp file")
        );
        let write_error = writer.write_all(b"more").await.unwrap_err();
        assert!(
            write_error
                .to_string()
                .contains("failed to prepare temp file")
        );
        assert!(!temp_file_path.exists());
        assert!(!file_path.exists());
    }

    #[tokio::test]
    async fn test_local_writer_drop_cleans_up() {
        let tmp = lance_core::utils::tempfile::TempStdDir::default();
        let file_path = tmp.join("test_drop.bin");
        let os_path = Path::from_absolute_path(&file_path).unwrap();
        let io_tracker = Arc::new(IOTracker::default());

        let named_temp = tempfile::NamedTempFile::new_in(&*tmp).unwrap();
        let temp_file_path = named_temp.path().to_owned();
        let (std_file, temp_path) = named_temp.into_parts();
        let file = tokio::fs::File::from_std(std_file);
        let mut writer = LocalWriter::new(file, os_path, temp_path, io_tracker);

        writer.write_all(b"some data").await.unwrap();
        assert!(temp_file_path.exists());

        // Drop without shutdown should clean up the temp file
        drop(writer);
        assert!(!temp_file_path.exists());
        assert!(!file_path.exists());
    }

    #[test]
    fn clamp_initial_upload_size_below_min_is_clamped_up() {
        assert_eq!(clamp_initial_upload_size(0), (INITIAL_UPLOAD_STEP, true));
        assert_eq!(
            clamp_initial_upload_size(INITIAL_UPLOAD_STEP - 1),
            (INITIAL_UPLOAD_STEP, true)
        );
    }

    #[test]
    fn clamp_initial_upload_size_within_range_is_unchanged() {
        assert_eq!(
            clamp_initial_upload_size(INITIAL_UPLOAD_STEP),
            (INITIAL_UPLOAD_STEP, false)
        );
        assert_eq!(
            clamp_initial_upload_size(MAX_UPLOAD_PART_SIZE),
            (MAX_UPLOAD_PART_SIZE, false)
        );
        let mid = INITIAL_UPLOAD_STEP * 8; // 40MB, in range
        assert_eq!(clamp_initial_upload_size(mid), (mid, false));
    }

    #[test]
    fn should_retry_upload_put_detects_transient_errors() {
        let request_timeout = OSError::Generic {
            store: "S3",
            source: Box::new(io::Error::other(
                "Server returned non-2xx status code: 400 Bad Request: \
                 <Error><Code>RequestTimeout</Code><Message>Your socket connection to the server \
                 was not read from or written to within the timeout period. Idle connections will \
                 be closed.</Message></Error>",
            )),
        };
        assert!(should_retry_upload_put(&request_timeout));

        let connection_reset = OSError::Generic {
            store: "S3",
            source: Box::new(io::Error::new(
                io::ErrorKind::ConnectionReset,
                "connection reset by peer",
            )),
        };
        assert!(should_retry_upload_put(&connection_reset));

        let not_retryable = OSError::Generic {
            store: "S3",
            source: Box::new(io::Error::other("access denied")),
        };
        assert!(!should_retry_upload_put(&not_retryable));
    }

    #[test]
    fn clamp_initial_upload_size_above_max_is_clamped_down() {
        assert_eq!(
            clamp_initial_upload_size(MAX_UPLOAD_PART_SIZE + 1),
            (MAX_UPLOAD_PART_SIZE, true)
        );
        assert_eq!(
            clamp_initial_upload_size(usize::MAX),
            (MAX_UPLOAD_PART_SIZE, true)
        );
    }

    /// Regression for the foot-gun where `LANCE_INITIAL_UPLOAD_SIZE=5368709120`
    /// (exactly 5 GiB, Pucheng's setting) caused a single-PUT of 5 GiB on
    /// shutdown — which S3 rejects with `EntityTooLarge`. After tightening
    /// `MAX_UPLOAD_PART_SIZE` to 5 GiB - 1, raw 5 GiB must clamp DOWN.
    #[test]
    fn clamp_initial_upload_size_at_5gib_clamps_down() {
        let exactly_5_gib: usize = 5 * 1024 * 1024 * 1024;
        assert_eq!(
            clamp_initial_upload_size(exactly_5_gib),
            (MAX_UPLOAD_PART_SIZE, true)
        );
    }
}
