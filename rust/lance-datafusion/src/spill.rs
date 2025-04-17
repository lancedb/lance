// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use arrow::ipc::{reader::StreamReader, writer::StreamWriter};
use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, Schema};
use datafusion::{
    execution::SendableRecordBatchStream, physical_plan::stream::RecordBatchStreamAdapter,
};
use datafusion_common::DataFusionError;
use futures::TryStreamExt;

/// A spill of Arrow data to a temporary file. The file is an Arrow IPC stream
/// file.
///
/// Use [`Self::write()`] to write batches to the spill. They will immediately
/// be flushed to the IPC stream file. The file is created on the first write.
/// Use [`Self::finish()`] once all batches have been written to finalize the
/// file.
///
/// To acquire a reading stream, call [`Self::read()`]. This can be called
/// before or after the spill has finished. If called before, the stream
/// will emit batches as they are written to the file. If called after, the stream
/// will emit all batches in the file. The stream will not complete until
/// [`Self::finish()`] is called.
///
/// When this is dropped, the temporary file is deleted. However, to handle
/// potential IO errors, it's preferable to call [`Self::shutdown()`] before
/// dropping the spill.
pub struct Spill {
    tmp_dir: tempfile::TempDir,
    schema: Arc<Schema>,
    path: PathBuf,
    state: SpillState,
    status_sender: tokio::sync::watch::Sender<WriteStatus>,
    status_receiver: tokio::sync::watch::Receiver<WriteStatus>,
}

enum SpillState {
    Uninitialized,
    Initialized {
        writer: AsyncStreamWriter,
        batches_written: usize,
    },
    Finished {
        batches_written: usize,
    },
}

#[derive(Clone, Copy, Debug)]
struct WriteStatus {
    finished: bool,
    batches_written: usize,
}

impl From<&SpillState> for WriteStatus {
    fn from(state: &SpillState) -> Self {
        let (finished, batches_written) = match state {
            SpillState::Uninitialized => (false, 0),
            SpillState::Initialized {
                batches_written, ..
            } => (false, *batches_written),
            SpillState::Finished { batches_written } => (true, *batches_written),
        };
        WriteStatus {
            finished,
            batches_written,
        }
    }
}

impl Spill {
    /// Creates a new spill writer. The temporary directory is created
    /// in the system's temporary directory. The schema is used to
    /// create the Arrow IPC stream file.
    pub fn new(tmp_dir: tempfile::TempDir, schema: Arc<Schema>) -> Self {
        let path = tmp_dir.path().join("spill.arrow");
        let initial_status = WriteStatus {
            finished: false,
            batches_written: 0,
        };
        let (status_sender, status_receiver) = tokio::sync::watch::channel(initial_status);
        Self {
            tmp_dir,
            schema,
            path,
            state: SpillState::Uninitialized,
            status_sender,
            status_receiver,
        }
    }

    /// Write a batch to the spill. The batch is immediately flushed to the
    /// IPC stream file.
    pub async fn write(&mut self, batch: RecordBatch) -> Result<(), DataFusionError> {
        if let SpillState::Finished { .. } = self.state {
            return Err(DataFusionError::Execution(
                "Spill has already been finished".to_string(),
            ));
        }

        let (writer, batches_written) = match &mut self.state {
            SpillState::Uninitialized => {
                let writer =
                    AsyncStreamWriter::open(self.path.clone(), self.schema.clone()).await?;
                self.state = SpillState::Initialized {
                    writer,
                    batches_written: 0,
                };
                if let SpillState::Initialized {
                    writer,
                    batches_written,
                } = &mut self.state
                {
                    (writer, batches_written)
                } else {
                    unreachable!()
                }
            }
            SpillState::Initialized {
                writer,
                batches_written,
            } => (writer, batches_written),
            _ => unreachable!(),
        };

        writer.write(batch).await?;
        *batches_written += 1;
        self.status_sender
            .send_replace(WriteStatus::from(&self.state));

        Ok(())
    }

    /// Complete the spill write. This will finalize the Arrow IPC stream file.
    /// The file will remain available for reading until [`Self::shutdown()`]
    /// or until the spill is dropped.
    pub async fn finish(&mut self) -> Result<(), DataFusionError> {
        // We create a temporary state to get an owned copy of current state.
        // Since we hold an exclusive reference to `self`, no one should be
        // able to see this temporary state.
        let tmp_state = SpillState::Finished { batches_written: 0 };
        match std::mem::replace(&mut self.state, tmp_state) {
            SpillState::Uninitialized => {
                return Err(DataFusionError::Execution(
                    "Spill has not been initialized".to_string(),
                ));
            }
            SpillState::Initialized {
                writer,
                batches_written,
            } => {
                writer.finish().await?;
                self.state = SpillState::Finished { batches_written };
                self.status_sender
                    .send_replace(WriteStatus::from(&self.state));
            }
            SpillState::Finished { .. } => {
                return Err(DataFusionError::Execution(
                    "Spill has already been finished".to_string(),
                ));
            }
        };

        Ok(())
    }

    /// Returns a stream of batches from the spill. The stream will emit
    /// batches as they are written to the spill. If the spill has already
    /// been finished, the stream will emit all batches in the spill.
    ///
    /// The stream will not complete until [`Self::finish()`] is called.
    ///
    /// If the spill has been dropped, an error will be returned.
    pub fn read(&self) -> Result<SendableRecordBatchStream, DataFusionError> {
        let mut rx = self.status_receiver.clone();
        let batches_read = 0;
        let path = self.path.clone();

        let stream = futures::stream::once(async move {
            if !rx.borrow().finished && rx.borrow().batches_written == 0 {
                // Wait for data to be written
                rx.wait_for(|status| status.finished || status.batches_written > 0)
                    .await
                    .map_err(|_| {
                        DataFusionError::Execution(
                            "Spill has been dropped before reader has finish.".into(),
                        )
                    })?;
            }

            let reader = AsyncStreamReader::open(path).await?;

            Ok::<_, DataFusionError>(futures::stream::try_unfold(
                (rx, batches_read, reader),
                move |(mut rx, mut batches_read, reader)| async move {
                    if !rx.borrow().finished && batches_read >= rx.borrow().batches_written {
                        // Wait for more data to be available
                        println!("waiting for more data");
                        rx.wait_for(|status| {
                            status.finished || status.batches_written > batches_read
                        })
                        .await
                        .map_err(|_| {
                            DataFusionError::Execution(
                                "Spill has been dropped before reader has finish.".into(),
                            )
                        })?;
                    } else if rx.borrow().finished && batches_read >= rx.borrow().batches_written {
                        return Ok(None);
                    }

                    let batch = reader.read().await?.ok_or_else(|| {
                        DataFusionError::Execution(
                            "Got fewer than expected batches from spill".into(),
                        )
                    })?;

                    batches_read += 1;

                    Ok(Some((batch, (rx, batches_read, reader))))
                },
            ))
        })
        .try_flatten();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            stream,
        )))
    }

    pub async fn shutdown(self) -> Result<(), DataFusionError> {
        let res =
            tokio::task::spawn_blocking(move || self.tmp_dir.close().map_err(ArrowError::from))
                .await;
        match res {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(DataFusionError::Execution(e.to_string())),
            Err(e) => Err(DataFusionError::Execution(e.to_string())),
        }
    }
}

/// An async wrapper around [`StreamWriter`]. Each call uses [`tokio::task::spawn_blocking`]
/// to spawn a blocking task to write the batch.
struct AsyncStreamWriter {
    writer: Arc<Mutex<StreamWriter<std::fs::File>>>,
}

impl AsyncStreamWriter {
    pub async fn open(path: PathBuf, schema: Arc<Schema>) -> Result<Self, ArrowError> {
        let writer = tokio::task::spawn_blocking(move || {
            let file = std::fs::File::create(&path).map_err(ArrowError::from)?;
            StreamWriter::try_new(file, &schema)
        })
        .await
        .unwrap()?;
        let writer = Arc::new(Mutex::new(writer));
        Ok(Self { writer })
    }

    pub async fn write(&self, batch: RecordBatch) -> Result<(), ArrowError> {
        let writer = self.writer.clone();
        tokio::task::spawn_blocking(move || {
            let mut writer = writer.lock().unwrap();
            writer.write(&batch)?;
            writer.flush()
        })
        .await
        .unwrap()
    }

    pub async fn finish(self) -> Result<(), ArrowError> {
        let writer = self.writer.clone();
        tokio::task::spawn_blocking(move || {
            let mut writer = writer.lock().unwrap();
            writer.finish()
        })
        .await
        .unwrap()
    }
}

struct AsyncStreamReader {
    reader: Arc<Mutex<StreamReader<std::fs::File>>>,
}

impl AsyncStreamReader {
    pub async fn open(path: PathBuf) -> Result<Self, ArrowError> {
        let reader = tokio::task::spawn_blocking(move || {
            let file = std::fs::File::open(&path).map_err(ArrowError::from)?;
            StreamReader::try_new(file, None)
        })
        .await
        .unwrap()?;
        let reader = Arc::new(Mutex::new(reader));
        Ok(Self { reader })
    }

    pub async fn read(&self) -> Result<Option<RecordBatch>, ArrowError> {
        let reader = self.reader.clone();
        tokio::task::spawn_blocking(move || {
            let mut reader = reader.lock().unwrap();
            reader.next()
        })
        .await
        .unwrap()
        .transpose()
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::Int32Array;
    use arrow_schema::{DataType, Field};
    use futures::{poll, StreamExt, TryStreamExt};

    use super::*;

    #[tokio::test]
    async fn test_spill() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let batches = [
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
            )
            .unwrap(),
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from(vec![4, 5, 6]))],
            )
            .unwrap(),
        ];

        // Create a stream
        let tmp_dir = tempfile::tempdir().unwrap();
        let mut spill = Spill::new(tmp_dir, schema.clone());

        // We can open a reader prior to writing any data. No batches will be ready.
        let mut stream_before = spill.read().unwrap();
        let mut stream_before_next = stream_before.next();
        let poll_res = poll!(&mut stream_before_next);
        assert!(poll_res.is_pending());

        // If we write a batch, the existing reader can now receive it.
        spill.write(batches[0].clone()).await.unwrap();
        let stream_before_batch1 = stream_before_next
            .await
            .expect("Expected a batch")
            .expect("Expected no error");
        assert_eq!(&stream_before_batch1, &batches[0]);
        let mut stream_before_next = stream_before.next();
        let poll_res = poll!(&mut stream_before_next);
        assert!(poll_res.is_pending());

        // We can also open a ready while the spill is being written to. We can
        // retrieve batches written so far immediately.
        let mut stream_during = spill.read().unwrap();
        let stream_during_batch1 = stream_during
            .next()
            .await
            .expect("Expected a batch")
            .expect("Expected no error");
        assert_eq!(&stream_during_batch1, &batches[0]);
        let mut stream_during_next = stream_during.next();
        let poll_res = poll!(&mut stream_during_next);
        assert!(poll_res.is_pending());

        // Once we finish the spill, readers can get remaining batches and will
        // reach the end of the stream.
        spill.write(batches[1].clone()).await.unwrap();
        spill.finish().await.unwrap();

        let stream_before_batch2 = stream_before_next
            .await
            .expect("Expected a batch")
            .expect("Expected no error");
        assert_eq!(&stream_before_batch2, &batches[1]);
        assert!(stream_before.next().await.is_none());

        let stream_during_batch2 = stream_during_next
            .await
            .expect("Expected a batch")
            .expect("Expected no error");
        assert_eq!(&stream_during_batch2, &batches[1]);
        assert!(stream_during.next().await.is_none());

        // Can also start a reader after finishing.
        let stream_after = spill.read().unwrap();
        let stream_after_batches = stream_after.try_collect::<Vec<_>>().await.unwrap();
        assert_eq!(&stream_after_batches, &batches);

        // Once we close the spill, the file is deleted.
        spill.shutdown().await.unwrap();
    }
}
