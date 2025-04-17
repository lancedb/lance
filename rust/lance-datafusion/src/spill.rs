// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    io::Error,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use arrow::ipc::{reader::StreamReader, writer::StreamWriter};
use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, Schema};
use datafusion::execution::SendableRecordBatchStream;
use datafusion_common::DataFusionError;

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

#[derive(Clone, Copy)]
struct WriteStatus {
    finished: bool,
    batches_written: usize,
}

impl From<&SpillState> for WriteStatus {
    fn from(state: &SpillState) -> Self {
        match state {
            SpillState::Uninitialized => WriteStatus {
                finished: false,
                batches_written: 0,
            },
            SpillState::Initialized {
                batches_written, ..
            } => WriteStatus {
                finished: false,
                batches_written: *batches_written,
            },
            SpillState::Finished { batches_written } => WriteStatus {
                finished: true,
                batches_written: *batches_written,
            },
        }
    }
}

impl Spill {
    /// Creates a new spill writer. The temporary directory is created
    /// in the system's temporary directory. The schema is used to
    /// create the Arrow IPC stream file.
    pub fn new(tmp_dir: tempfile::TempDir, schema: Arc<Schema>) -> Self {
        let path = tmp_dir.path().join("spill.arrow");
        let (status_sender, status_receiver) = tokio::sync::watch::channel(WriteStatus {
            finished: false,
            batches_written: 0,
        });
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
        match &mut self.state {
            SpillState::Uninitialized => {
                let writer =
                    AsyncStreamWriter::open(self.path.clone(), self.schema.clone()).await?;
                writer.write(batch).await?;
                self.state = SpillState::Initialized {
                    writer,
                    batches_written: 1,
                };
                self.status_sender
                    .send_replace(WriteStatus::from(&self.state));
            }
            SpillState::Initialized {
                writer,
                batches_written,
            } => {
                writer.write(batch).await?;
                *batches_written += 1;
                self.status_sender
                    .send_replace(WriteStatus::from(&self.state));
            }
            SpillState::Finished { .. } => {
                return Err(DataFusionError::Execution(
                    "Spill has already been finished".to_string(),
                ));
            }
        }

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

    pub fn read(&self) -> Result<SendableRecordBatchStream, DataFusionError> {
        todo!()
        // Take a copy of the write path, status reciever.
        // In thee stream, watch the status reciever to show > 0 batches written
        // before opening the file.
        // After that, open the file and create a stream reader.
        // Wait until the status reciever says a batch has been written, before
        // reading the next batch.
        // When the status reciever says the spill has been finished, read the
        // remaining batches in the file and then finish the stream.
    }

    pub async fn shutdown(self) -> Result<(), DataFusionError> {
        self.tmp_dir.close()?;
        Ok(())
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
            writer.write(&batch)
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
    use super::*;

    #[tokio::test]
    async fn test_spill() {
        // Create a stream

        // Open a reader before writing. Assert we can't get any data.

        // Write some data. Assert it shows up, but can't advanced past.

        // Open a reader now. Assert we can get the data.

        // Finish the spill. Assert we can get all the data from two existing readers.

        // Create a new reader. Assert we can get all the data from it.

        // Close the spill. Assert new readers can't get any data.

        // Assert the file is actually deleted.
    }
}
