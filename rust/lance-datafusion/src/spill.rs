// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    io::{BufReader, BufWriter},
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
use lance_arrow::memory::MemoryAccumulator;
use lance_core::error::LanceOptionExt;

/// Start a spill of Arrow data to a temporary file. The file is an Arrow IPC
/// stream file.
///
/// Up to `memory_limit` bytes of data can be buffered in memory before a spill
/// is created. If the memory limit is never reached before [`SpillSender::finish()`]
/// is called, then the data will simply be kept in memory and no spill will be
/// created.
///
/// The [`SpillSender`] allows you to write batches to the spill.
///
/// The [`SpillReceiver`] can open a [`SendableRecordBatchStream`] that reads
/// batches from the spill. This can be opened before, during, or after batches
/// have been written to the spill.
///
/// Once [`SpillSender`] is dropped, the temporary file is deleted. This will
/// cause the [`SpillReceiver`] to return an error if it is still open.
pub fn create_spill(
    path: std::path::PathBuf,
    schema: Arc<Schema>,
    memory_limit: usize,
) -> (SpillSender, SpillReceiver) {
    let initial_status = WriteStatus::default();
    let (status_sender, status_receiver) = tokio::sync::watch::channel(initial_status);
    let sender = SpillSender {
        memory_limit,
        path: path.clone(),
        schema: schema.clone(),
        state: SpillState::default(),
        status_sender,
    };

    let receiver = SpillReceiver {
        status_receiver,
        path,
        schema,
    };

    (sender, receiver)
}

#[derive(Clone)]
pub struct SpillReceiver {
    status_receiver: tokio::sync::watch::Receiver<WriteStatus>,
    path: PathBuf,
    schema: Arc<Schema>,
}

impl SpillReceiver {
    /// Returns a stream of batches from the spill. The stream will emit
    /// batches as they are written to the spill. If the spill has already
    /// been finished, the stream will emit all batches in the spill.
    ///
    /// The stream will not complete until [`Self::finish()`] is called.
    ///
    /// If the spill has been dropped, an error will be returned.
    pub fn read(&self) -> SendableRecordBatchStream {
        let rx = self.status_receiver.clone();
        let reader = SpillReader::new(rx, self.path.clone());

        let stream = futures::stream::try_unfold(reader, move |mut reader| async move {
            match reader.read().await {
                Ok(None) => Ok(None),
                Ok(Some(batch)) => Ok(Some((batch, reader))),
                Err(err) => Err(err),
            }
        });

        Box::pin(RecordBatchStreamAdapter::new(self.schema.clone(), stream))
    }
}

struct SpillReader {
    pub batches_read: usize,
    receiver: tokio::sync::watch::Receiver<WriteStatus>,
    state: SpillReaderState,
}

enum SpillReaderState {
    Buffered { spill_path: PathBuf },
    Reader { reader: AsyncStreamReader },
}

impl SpillReader {
    fn new(receiver: tokio::sync::watch::Receiver<WriteStatus>, spill_path: PathBuf) -> Self {
        Self {
            batches_read: 0,
            receiver,
            state: SpillReaderState::Buffered { spill_path },
        }
    }

    async fn wait_for_more_data(&mut self) -> Result<(), DataFusionError> {
        let status = self
            .receiver
            .wait_for(|status| {
                status.error.is_some()
                    || status.finished
                    || status.batches_written() > self.batches_read
            })
            .await
            .map_err(|_| {
                DataFusionError::Execution(
                    "Spill has been dropped before reader has finish.".into(),
                )
            })?;

        if let Some(error) = &status.error {
            let mut guard = error.lock().ok().expect_ok()?;
            return Err(DataFusionError::from(&mut (*guard)));
        }

        Ok(())
    }

    async fn upgrade(&mut self) -> Result<(), ArrowError> {
        if let SpillReaderState::Buffered { spill_path } = &self.state {
            let reader = AsyncStreamReader::open(spill_path.clone()).await?;
            // Skip batches we've already read.
            for _ in 0..self.batches_read {
                reader.read().await?;
            }
            self.state = SpillReaderState::Reader { reader };
        }
        Ok(())
    }

    async fn read(&mut self) -> Result<Option<RecordBatch>, DataFusionError> {
        self.wait_for_more_data().await?;

        if let SpillReaderState::Buffered { .. } = &self.state {
            self.upgrade().await?;
        }

        match self.state {
            SpillReaderState::Buffered { .. } => {
                if let DataLocation::Buffered { batches } = &self.receiver.borrow().data_location {
                    if self.batches_read < batches.len() {
                        let batch = batches[self.batches_read].clone();
                        self.batches_read += 1;
                        Ok(Some(batch))
                    } else {
                        Ok(None)
                    }
                } else {
                    unreachable!("maybe")
                }
            }
            SpillReaderState::Reader { ref mut reader } => {
                let batch = reader.read().await?;
                if batch.is_some() {
                    self.batches_read += 1;
                }
                Ok(batch)
            }
        }
    }
}

/// A spill of Arrow data to an IPC stream file.
///
/// Use [`Self::write()`] to write batches to the spill. They will immediately
/// be flushed to the IPC stream file. The file is created on the first write.
/// Use [`Self::finish()`] once all batches have been written to finalize the
/// file.
pub struct SpillSender {
    memory_limit: usize,
    schema: Arc<Schema>,
    path: PathBuf,
    state: SpillState,
    status_sender: tokio::sync::watch::Sender<WriteStatus>,
}

enum SpillState {
    Buffering {
        batches: Vec<RecordBatch>,
        memory_accumulator: MemoryAccumulator,
    },
    Spilling {
        writer: AsyncStreamWriter,
        batches_written: usize,
    },
    Finished {
        batches: Option<Arc<[RecordBatch]>>,
        batches_written: usize,
    },
    Errored {
        error: Arc<Mutex<SpillError>>,
    },
}

impl Default for SpillState {
    fn default() -> Self {
        Self::Buffering {
            batches: Vec::new(),
            memory_accumulator: MemoryAccumulator::default(),
        }
    }
}

#[derive(Clone, Debug, Default)]
struct WriteStatus {
    error: Option<Arc<Mutex<SpillError>>>,
    finished: bool,
    data_location: DataLocation,
}

impl WriteStatus {
    fn batches_written(&self) -> usize {
        match &self.data_location {
            DataLocation::Buffered { batches } => batches.len(),
            DataLocation::Spilled {
                batches_written, ..
            } => *batches_written,
        }
    }
}

#[derive(Clone, Debug)]
enum DataLocation {
    Buffered { batches: Arc<[RecordBatch]> },
    Spilled { batches_written: usize },
}

impl Default for DataLocation {
    fn default() -> Self {
        Self::Buffered {
            batches: Arc::new([]),
        }
    }
}

/// A DataFusion error that be be emitted multiple times. We provide the
/// Original error first, and subsequent conversions provide a copy with a
/// string representation of the original error.
#[derive(Debug)]
enum SpillError {
    Original(DataFusionError),
    Copy(DataFusionError),
}

impl From<DataFusionError> for SpillError {
    fn from(err: DataFusionError) -> Self {
        Self::Original(err)
    }
}

impl From<&mut SpillError> for DataFusionError {
    fn from(err: &mut SpillError) -> Self {
        match err {
            SpillError::Original(inner) => {
                let copy = Self::Execution(inner.to_string());
                let original = std::mem::replace(err, SpillError::Copy(copy));
                if let SpillError::Original(inner) = original {
                    inner
                } else {
                    unreachable!()
                }
            }
            SpillError::Copy(Self::Execution(message)) => Self::Execution(message.clone()),
            _ => unreachable!(),
        }
    }
}

impl From<&SpillState> for WriteStatus {
    fn from(state: &SpillState) -> Self {
        match state {
            SpillState::Buffering { batches, .. } => Self {
                finished: false,
                data_location: DataLocation::Buffered {
                    batches: batches.clone().into(),
                },
                error: None,
            },
            SpillState::Spilling {
                batches_written, ..
            } => Self {
                finished: false,
                data_location: DataLocation::Spilled {
                    batches_written: *batches_written,
                },
                error: None,
            },
            SpillState::Finished {
                batches_written,
                batches,
            } => {
                let data_location = if let Some(batches) = batches {
                    DataLocation::Buffered {
                        batches: batches.clone(),
                    }
                } else {
                    DataLocation::Spilled {
                        batches_written: *batches_written,
                    }
                };
                Self {
                    finished: true,
                    data_location,
                    error: None,
                }
            }
            SpillState::Errored { error } => Self {
                finished: true,
                data_location: DataLocation::default(), // Doesn't matter.
                error: Some(error.clone()),
            },
        }
    }
}

impl SpillSender {
    /// Write a batch to the spill. The batch is immediately flushed to the
    /// IPC stream file.
    pub async fn write(&mut self, batch: RecordBatch) -> Result<(), DataFusionError> {
        if let SpillState::Finished { .. } = self.state {
            return Err(DataFusionError::Execution(
                "Spill has already been finished".to_string(),
            ));
        }

        if let SpillState::Errored { .. } = &self.state {
            return Err(DataFusionError::Execution(
                "Spill has sent an error".to_string(),
            ));
        }

        let (writer, batches_written) = match &mut self.state {
            SpillState::Buffering {
                batches,
                ref mut memory_accumulator,
            } => {
                memory_accumulator.record_batch(&batch);

                if memory_accumulator.total() > self.memory_limit {
                    let writer =
                        AsyncStreamWriter::open(self.path.clone(), self.schema.clone()).await?;
                    for batch in batches.drain(..) {
                        writer.write(batch).await?;
                    }
                    self.state = SpillState::Spilling {
                        writer,
                        batches_written: batches.len(),
                    };
                    if let SpillState::Spilling {
                        writer,
                        batches_written,
                    } = &mut self.state
                    {
                        (writer, batches_written)
                    } else {
                        unreachable!()
                    }
                } else {
                    batches.push(batch);
                    self.status_sender
                        .send_replace(WriteStatus::from(&self.state));
                    return Ok(());
                }
            }
            SpillState::Spilling {
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

    /// Send an error to the spill. This will be sent to all readers of the
    /// spill.
    pub fn send_error(&mut self, err: DataFusionError) {
        let error = Arc::new(Mutex::new(err.into()));
        self.state = SpillState::Errored { error };
        self.status_sender
            .send_replace(WriteStatus::from(&self.state));
    }

    /// Complete the spill write. This will finalize the Arrow IPC stream file.
    /// The file will remain available for reading until [`Self::shutdown()`]
    /// or until the spill is dropped.
    pub async fn finish(&mut self) -> Result<(), DataFusionError> {
        // We create a temporary state to get an owned copy of current state.
        // Since we hold an exclusive reference to `self`, no one should be
        // able to see this temporary state.
        let tmp_state = SpillState::Finished {
            batches_written: 0,
            batches: None,
        };
        match std::mem::replace(&mut self.state, tmp_state) {
            SpillState::Buffering { batches, .. } => {
                let batches_written = batches.len();
                self.state = SpillState::Finished {
                    batches_written,
                    batches: Some(batches.into()),
                };
                self.status_sender
                    .send_replace(WriteStatus::from(&self.state));
            }
            SpillState::Spilling {
                writer,
                batches_written,
            } => {
                writer.finish().await?;
                self.state = SpillState::Finished {
                    batches_written,
                    batches: None,
                };
                self.status_sender
                    .send_replace(WriteStatus::from(&self.state));
            }
            SpillState::Finished { .. } => {
                return Err(DataFusionError::Execution(
                    "Spill has already been finished".to_string(),
                ));
            }
            SpillState::Errored { .. } => {
                return Err(DataFusionError::Execution(
                    "Spill has sent an error".to_string(),
                ));
            }
        };

        Ok(())
    }
}

/// An async wrapper around [`StreamWriter`]. Each call uses [`tokio::task::spawn_blocking`]
/// to spawn a blocking task to write the batch.
struct AsyncStreamWriter {
    writer: Arc<Mutex<StreamWriter<BufWriter<std::fs::File>>>>,
}

impl AsyncStreamWriter {
    pub async fn open(path: PathBuf, schema: Arc<Schema>) -> Result<Self, ArrowError> {
        let writer = tokio::task::spawn_blocking(move || {
            let file = std::fs::File::create(&path).map_err(ArrowError::from)?;
            let writer = BufWriter::new(file);
            StreamWriter::try_new(writer, &schema)
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
    reader: Arc<Mutex<StreamReader<BufReader<std::fs::File>>>>,
}

impl AsyncStreamReader {
    pub async fn open(path: PathBuf) -> Result<Self, ArrowError> {
        let reader = tokio::task::spawn_blocking(move || {
            let file = std::fs::File::open(&path).map_err(ArrowError::from)?;
            let reader = BufReader::new(file);
            StreamReader::try_new(reader, None)
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
        let path = tmp_dir.path().join("spill.arrows");
        let (mut spill, receiver) = create_spill(path.clone(), schema.clone(), 0);

        // We can open a reader prior to writing any data. No batches will be ready.
        let mut stream_before = receiver.read();
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
        let mut stream_during = receiver.read();
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
        let stream_after = receiver.read();
        let stream_after_batches = stream_after.try_collect::<Vec<_>>().await.unwrap();
        assert_eq!(&stream_after_batches, &batches);

        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn test_spill_error() {
        // Create a spill
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join("spill.arrows");
        let (mut spill, receiver) = create_spill(path.clone(), schema.clone(), 0);
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();

        spill.write(batch.clone()).await.unwrap();

        let mut stream = receiver.read();
        let stream_batch = stream
            .next()
            .await
            .expect("Expected a batch")
            .expect("Expected no error");
        assert_eq!(&stream_batch, &batch);

        spill.send_error(DataFusionError::ResourcesExhausted("🥱".into()));
        let stream_error = stream
            .next()
            .await
            .expect("Expected an error")
            .expect_err("Expected an error");
        assert!(matches!(
            stream_error,
            DataFusionError::ResourcesExhausted(message) if message == "🥱"
        ));

        // If we try to write after sending an error, it should return an error.
        let err = spill.write(batch).await;
        assert!(matches!(
            err,
            Err(DataFusionError::Execution(message)) if message == "Spill has sent an error"
        ));

        // If we try to finish after sending an error, it should return an error.
        let err = spill.finish().await;
        assert!(matches!(
            err,
            Err(DataFusionError::Execution(message)) if message == "Spill has sent an error"
        ));

        // If we try to read after sending an error, it should return an error.
        let mut stream = receiver.read();
        let stream_error = stream
            .next()
            .await
            .expect("Expected an error")
            .expect_err("Expected an error");
        assert!(matches!(
            stream_error,
            DataFusionError::Execution(message) if message.contains("🥱")
        ));

        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn test_spill_buffered() {
        todo!("test spill with less than memory limit.");

        todo!("test spill starts under memory limit");
    }
}
