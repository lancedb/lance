// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, SchemaRef};
use futures::{lock::Mutex, stream::StreamExt};
use tokio::sync::mpsc;

use lance::dataset::scanner::{DatasetRecordBatchStream, Scanner as LanceScanner};
use lance_io::stream::RecordBatchStream;

use crate::rt;

const READER_CHANNEL_CAPACITY: usize = 2;

enum ReaderMessage {
    Batch(Result<RecordBatch, ArrowError>),
    Finished,
}

/// Lance's RecordBatchReader
///
/// The async scan is driven by one background producer for the lifetime of the
/// reader. The synchronous Arrow C stream consumer receives batches through a
/// bounded channel, avoiding a runtime task spawn and cross-thread rendezvous
/// for every batch while preserving backpressure.
pub struct LanceReader {
    schema: SchemaRef,
    receiver: std::sync::Arc<Mutex<mpsc::Receiver<ReaderMessage>>>,
    finished: bool,
}

impl LanceReader {
    pub async fn try_new(mut scanner: std::sync::Arc<LanceScanner>) -> ::lance::Result<Self> {
        let stream = std::sync::Arc::make_mut(&mut scanner)
            .try_into_stream()
            .await?;
        Ok(Self::from_stream(stream))
    }

    pub fn from_stream(mut stream: DatasetRecordBatchStream) -> Self {
        let schema = stream.schema();
        let (sender, receiver) = mpsc::channel(READER_CHANNEL_CAPACITY);
        rt().spawn_background(None, async move {
            loop {
                let next = tokio::select! {
                    _ = sender.closed() => break,
                    next = stream.next() => next,
                };
                let Some(batch) = next else {
                    let _ = sender.send(ReaderMessage::Finished).await;
                    break;
                };
                if sender
                    .send(ReaderMessage::Batch(batch.map_err(ArrowError::from)))
                    .await
                    .is_err()
                {
                    break;
                }
            }
        });
        Self {
            schema,
            receiver: std::sync::Arc::new(Mutex::new(receiver)),
            finished: false,
        }
    }
}

impl Iterator for LanceReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        let receiver = self.receiver.clone();
        let recv = async move { receiver.lock().await.recv().await };
        let result = if tokio::runtime::Handle::try_current().is_ok() {
            // Blocking this thread's active Tokio runtime can panic or deadlock.
            // Keep the previous cross-thread rendezvous for this uncommon path.
            rt().spawn(None, recv)
        } else {
            rt().block_on(None, recv)
        };
        match result {
            Ok(Some(ReaderMessage::Batch(batch))) => Some(batch),
            Ok(Some(ReaderMessage::Finished)) => {
                self.finished = true;
                None
            }
            Ok(None) => {
                self.finished = true;
                Some(Err(ArrowError::ExternalError(Box::new(
                    std::io::Error::other("Lance reader producer terminated before end of stream"),
                ))))
            }
            Err(err) => Some(Err(ArrowError::ExternalError(Box::new(err)))),
        }
    }
}

impl RecordBatchReader for LanceReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, mpsc::Sender},
        time::Duration,
    };

    use arrow_array::{Int32Array, RecordBatchReader, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::{
        error::DataFusionError,
        physical_plan::{SendableRecordBatchStream, stream::RecordBatchStreamAdapter},
    };
    use futures::stream;

    use super::*;

    fn make_reader(
        schema: SchemaRef,
        batches: impl futures::Stream<Item = Result<RecordBatch, DataFusionError>> + Send + 'static,
    ) -> LanceReader {
        let stream: SendableRecordBatchStream =
            Box::pin(RecordBatchStreamAdapter::new(schema, batches));
        LanceReader::from_stream(DatasetRecordBatchStream::new(stream))
    }

    #[test]
    fn test_reader_preserves_schema_batches_and_end_of_stream() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let expected = (0..3)
            .map(|batch_index| {
                RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(Int32Array::from(vec![
                        batch_index * 2,
                        batch_index * 2 + 1,
                    ]))],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let batches = stream::iter(expected.clone().into_iter().map(Ok));
        let mut reader = make_reader(schema.clone(), batches);

        assert_eq!(reader.schema(), schema);
        let actual = reader.by_ref().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(actual, expected);
        assert!(reader.next().is_none());
    }

    #[test]
    fn test_reader_propagates_stream_errors() {
        let schema = Arc::new(Schema::empty());
        let batches = stream::iter([Err(DataFusionError::Execution(
            "expected reader error".to_string(),
        ))]);
        let mut reader = make_reader(schema, batches);

        let error = reader.next().unwrap().unwrap_err();
        assert!(error.to_string().contains("expected reader error"));
        assert!(reader.next().is_none());
    }

    #[test]
    fn test_reader_reports_producer_panic_after_a_batch() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let expected = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let batches = stream::iter([Ok(expected.clone())]).chain(stream::poll_fn(|_| {
            panic!("expected producer panic");
        }));
        let mut reader = make_reader(schema, batches);

        assert_eq!(reader.next().unwrap().unwrap(), expected);
        let error = reader.next().unwrap().unwrap_err();
        assert!(
            error
                .to_string()
                .contains("producer terminated before end of stream")
        );
        assert!(reader.next().is_none());
    }

    struct DropNotify(Sender<()>);

    impl Drop for DropNotify {
        fn drop(&mut self) {
            self.0.send(()).ok();
        }
    }

    #[test]
    fn test_reader_drop_cancels_pending_stream() {
        let schema = Arc::new(Schema::empty());
        let (drop_sender, drop_receiver) = std::sync::mpsc::channel();
        let (started_sender, started_receiver) = std::sync::mpsc::channel();
        let batches = stream::once(async move {
            let _drop_notify = DropNotify(drop_sender);
            started_sender.send(()).ok();
            std::future::pending::<Result<RecordBatch, DataFusionError>>().await
        });
        let reader = make_reader(schema, batches);

        started_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("the producer should poll the stream");
        drop(reader);
        drop_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("dropping the reader should cancel and drop the producer stream");
    }

    #[test]
    fn test_reader_bounds_wide_batch_read_ahead() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "payload",
            DataType::Utf8,
            false,
        )]));
        let (poll_sender, poll_receiver) = std::sync::mpsc::channel();
        let batch_schema = schema.clone();
        let batches = stream::unfold(0, move |batch_index| {
            let poll_sender = poll_sender.clone();
            let batch_schema = batch_schema.clone();
            async move {
                if batch_index == 10 {
                    return None;
                }
                poll_sender.send(batch_index).ok();
                let batch = RecordBatch::try_new(
                    batch_schema,
                    vec![Arc::new(StringArray::from_iter_values(std::iter::once(
                        "x".repeat(1024 * 1024),
                    )))],
                )
                .unwrap();
                Some((Ok(batch), batch_index + 1))
            }
        });
        let mut reader = make_reader(schema, batches);

        assert_eq!(
            (0..3)
                .map(|_| poll_receiver.recv_timeout(Duration::from_secs(1)).unwrap())
                .collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert!(
            poll_receiver
                .recv_timeout(Duration::from_millis(100))
                .is_err()
        );

        assert_eq!(reader.next().unwrap().unwrap().num_rows(), 1);
        assert_eq!(
            poll_receiver.recv_timeout(Duration::from_secs(1)).unwrap(),
            3
        );
    }

    #[test]
    fn test_reader_can_be_consumed_from_background_runtime() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let expected = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let batches = stream::iter([Ok(expected.clone())]);
        let mut reader = make_reader(schema, batches);

        let actual = rt()
            .spawn(None, async move { reader.next().unwrap().unwrap() })
            .unwrap();
        assert_eq!(actual, expected);
    }
}
