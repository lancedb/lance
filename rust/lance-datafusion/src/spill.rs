// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{path::PathBuf, sync::Arc};

use arrow_array::RecordBatch;
use arrow_schema::ArrowError;
use datafusion::physical_plan::common::IPCWriter;
use datafusion_common::DataFusionError;
use lance_core::datatypes::Schema;

/// A spill writer that writes to a temporary file.
pub struct SpillWriter {
    tmp_dir: tempfile::TempDir,
    path: PathBuf,
    writer: IPCWriter,
}

impl SpillWriter {
    pub async fn try_new(schema: &Schema) -> Result<Self, DataFusionError> {
        tokio::task::spawn_blocking(|| {
            let tmp_dir = tempfile::tempdir()?;
            let path = tmp_dir.path().join("spill.arrows");
            let writer = IPCWriter::new(&path, schema)?;
            Ok(SpillWriter {
                tmp_dir,
                path,
                writer,
            })
        })?
    }

    pub async fn write(&mut self, batch: &RecordBatch) -> Result<(), DataFusionError> {
        tokio::task::spawn_blocking(move || {
            self.writer.write(batch)?;
            Ok(())
        })
        .await?
    }

    pub async fn finish(self) -> Result<Spill, DataFusionError> {
        let schema = self.writer.schema().clone();
        tokio::task::spawn_blocking(move || self.writer.finish()).await??;
        let tmp_dir = Arc::new(self.tmp_dir);
        let path = Arc::new(self.path);
        Ok(Spill {
            tmp_dir,
            path,
            schema: Arc::new(schema),
        })
    }
}

#[derive(Clone)]
pub struct Spill {
    tmp_dir: Arc<tempfile::TempDir>,
    path: Arc<PathBuf>,
    schema: Arc<Schema>,
}

impl Spill {
    pub fn read(&self) -> Result<SpillReader, ArrowError> {
        todo!()
    }
}

pub struct SpillReader {
    tmp_dir: Arc<tempfile::TempDir>,
}

impl RecordBatchStream for SpillReader {
    fn schema(&self) -> &Schema {
        todo!()
    }
}

impl Stream for SpillReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Option<Self::Item>> {
        todo!()
    }
}

// #[derive(Clone)]
// pub struct SpilledStream {
//     tmp_dir: Arc<tempfile::TempDir>,
//     path: Arc<PathBuf>,
//     schema: Arc<ArrowSchema>,
// }

// impl SpilledStream {
//     #[instrument(level = "debug", skip_all, fields(
//         num_rows = tracing::field::Empty,
//         num_batches = tracing::field::Empty,
//         num_bytes = tracing::field::Empty,
//     ))]
//     async fn try_new(mut stream: SendableRecordBatchStream) -> Result<Self> {
//         let tmp_dir = tempfile::tempdir()?;
//         let path = tmp_dir.path().join("spill.arrows");
//         let schema = stream.schema();

//         // We don't split the batches up. We assume if we need to read in smaller
//         // increments, the writer has already handled this.

//         // Writing an IPC file is synchronous, so we spawn a blocking task to do it.
//         let (tx, mut rx) = tokio::sync::mpsc::channel(0);
//         let schema_ref = schema.clone();
//         let writer_fut = tokio::task::spawn_blocking(move || {
//             let mut writer = IPCWriter::new(&path, schema_ref.as_ref())?;
//             while let Some(batch) = rx.blocking_recv() {
//                 if let Err(err) = writer.write(&batch) {
//                     return Err(err);
//                 }
//             }
//             Ok((writer, path))
//         });

//         tokio::pin!(writer_fut);

//         loop {
//             tokio::select! {
//                 res = stream.next() => {
//                     match res {
//                         Some(Ok(batch)) => {
//                             tx.send(batch).await.map_err(|_| Error::Internal {
//                                 message: "Failed to send batch to writer".into(),
//                                 location: location!(),
//                             })?;
//                         }
//                         Some(Err(err)) => {
//                             writer_fut.abort();
//                             // Delete the tmp dir in the background so we don't block.
//                             tokio::task::spawn_blocking(move || drop(tmp_dir));
//                             return Err(err.into());
//                         },
//                         None => {
//                             // No more batches, so we can close the writer
//                             drop(tx);
//                             let (writer, path) = writer_fut.await??;
//                             let current_span = tracing::Span::current();
//                             current_span.record("num_rows", &writer.num_rows);
//                             current_span.record("num_batches", &writer.num_batches);
//                             current_span.record("num_bytes", &writer.num_bytes);
//                             return Ok(SpilledStream {
//                                 tmp_dir: Arc::new(tmp_dir),
//                                 path: Arc::new(path),
//                                 schema,
//                             });
//                         }
//                     }
//                 },
//                 res = &mut writer_fut => {
//                     // Writer task finished before stream finished, so we need to delete the tmp dir
//                     let _ = tokio::task::spawn_blocking(move || drop(tmp_dir));
//                     match res {
//                         Ok(Ok(_)) => {
//                             return Err(Error::Internal {
//                                 message: "Writer finished before stream finished".into(),
//                                 location: location!(),
//                             });
//                         }
//                         Ok(Err(err)) => {
//                             return Err(err.into());
//                         }
//                         Err(_) => {
//                             return Err(Error::Internal {
//                                 message: "Writer task was cancelled".into(),
//                                 location: location!(),
//                             });
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     fn get_stream(&self) -> SendableRecordBatchStream {
//         // IPC reader is a blocking operation, so we spawn a blocking task to do it.
//         let self_clone = self.clone();
//         let stream: Pin<Box<dyn Stream<Item = DFResult<RecordBatch>> + Send>> = Box::pin(futures::stream::once(async move {
//             tokio::task::spawn_blocking(move || {
//                 let file = std::fs::File::open(self_clone.path.as_ref())?;
//                 arrow_ipc::reader::StreamReader::try_new_buffered(file, None)
//             }).await
//         }).flat_map(|reader| {
//             match reader {
//                 Ok(Ok(reader)) => {
//                     Box::pin(futures::stream::try_unfold(reader, |mut reader| async move {
//                         let fut = tokio::task::spawn_blocking(move || {
//                             let batch = reader.next();
//                             (batch, reader)
//                         });
//                         match fut.await {
//                             Ok((Some(Ok(batch)), reader)) => Ok(Some((batch, reader))),
//                             Ok((Some(Err(err)), _)) => Err(err.into()),
//                             Ok((None, _)) => Ok(None),
//                             Err(err) => Err(DataFusionError::ExecutionJoin(err)),
//                         }
//                     })) as Pin<Box<dyn Stream<Item = DFResult<RecordBatch>> + Send>>
//                 }
//                 Ok(Err(err)) => {
//                     Box::pin(futures::stream::once(futures::future::ready(Err(err.into()))))
//                 }
//                 Err(err) => {
//                     Box::pin(futures::stream::once(futures::future::ready(Err(
//                         DataFusionError::ExecutionJoin(err)))))
//                 }
//             }
//         }));

//         Box::pin(RecordBatchStreamAdapter::new(
//             self.schema.clone(),
//             stream,
//         ))
//     }
// }

// struct SpilledStreamReader {
//     spill: SpilledStream,
//     reader: StreamRearder
// }
