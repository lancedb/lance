// Copyright 2024 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use bytes::Bytes;
use futures::channel::oneshot;
use futures::{FutureExt, Stream, StreamExt};
use object_store::path::Path;
use snafu::{location, Location};
use std::future::Future;
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::task::Poll;
use tokio::sync::mpsc;

use lance_core::{Error, Result};

use crate::object_store::ObjectStore;
use crate::traits::Reader;

type FollowUpTask<C> = Box<dyn FnOnce(C, Vec<Bytes>) -> BatchRequest<C> + Send>;

/// A collection of requested I/O operations with an optional follow-up task
pub struct BatchRequest<C: Send> {
    iops: Vec<Range<u64>>,
    context: C,
    follow_up: Option<FollowUpTask<C>>,
}

impl BatchRequest<()> {
    /// Create a new simple request with no context or follow-up
    ///
    /// # Arguments
    ///
    /// * iops - the ranges to request.  There will be one `Bytes`
    ///   instance in the response for each range (even if some
    ///   ranges are coalesced).  For best performance these should
    ///   be sorted so that the end of each range is less than or
    ///   equal to the start of the next range
    pub fn new_simple(iops: Vec<Range<u64>>) -> Self {
        Self {
            iops,
            context: (),
            follow_up: None,
        }
    }
}

impl<C: Send> BatchRequest<C> {
    /// Create a new request with context
    ///
    /// The context will be delivered with the response.
    ///
    /// # Arguments
    ///
    /// * iops - the ranges to request.  See [`Self::new_simple`] for
    ///   more details
    /// * context - a context object.  This will be delivered with the
    ///   response.
    pub fn new_with_context(iops: Vec<Range<u64>>, context: C) -> Self {
        Self {
            iops,
            context,
            follow_up: None,
        }
    }

    /// Create a new request with context and a follow-up
    ///
    /// The context will be delivered to the follow-up task and, eventually,
    /// to the response.
    ///
    /// The follow-up task will be given the loaded data and then is expected
    /// to generate an additional request.  For example, this is used when
    /// doing an indirect load where the offsets or sizes of the desired data
    /// are stored on disk.
    ///
    /// # Arguments
    ///
    /// * iops - the ranges to request.  See [`Self::new_simple`] for
    ///   more details
    /// * context - a context object.  This will be delivered with the
    ///   response.
    /// * follow_up - a follow up task that will generate the next chunk
    ///   of I/O operations
    pub fn new_with_follow_up<F: FnOnce(C, Vec<Bytes>) -> Self + Send + 'static>(
        iops: Vec<Range<u64>>,
        context: C,
        follow_up: F,
    ) -> Self {
        Self {
            iops,
            context,
            follow_up: Some(Box::new(follow_up)),
        }
    }
}

// A batch of data loaded in response to a BatchRequest
pub struct LoadedBatch<C: Send> {
    /// The loaded data, grouped into one or more data buffers
    ///
    /// Each requested Range will result in exactly one `Bytes` object in this
    /// result.  This is true even if the ranges were coalesced into a single
    /// read by the scheduler.
    pub data_buffers: Vec<Bytes>,
    /// The context object that was provided with the original request.
    pub context: C,
}

// There is one instance of MutableBatch shared by all the I/O operations
// that make up a single BatchRequest.  When all the I/O operations complete
// then the MutableBatch goes out of scope and the batch request is considered
// complete
struct MutableBatch<F: FnOnce(Result<Vec<Bytes>>) + Send> {
    when_done: Option<F>,
    data_buffers: Vec<Bytes>,
    err: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
}

impl<F: FnOnce(Result<Vec<Bytes>>) + Send> MutableBatch<F> {
    fn new(when_done: F, num_data_buffers: u32) -> Self {
        Self {
            when_done: Some(when_done),
            data_buffers: vec![Bytes::default(); num_data_buffers as usize],
            err: None,
        }
    }
}

// Rather than keep track of when all the I/O requests are finished so that we
// can deliver the batch of data we let Rust do that for us.  When all I/O's are
// done then the MutableBatch will go out of scope and we know we have all the
// data.
impl<F: FnOnce(Result<Vec<Bytes>>) + Send> Drop for MutableBatch<F> {
    fn drop(&mut self) {
        // If we have an error, return that.  Otherwise return the data
        let result = if self.err.is_some() {
            Err(Error::Wrapped {
                error: self.err.take().unwrap(),
                location: location!(),
            })
        } else {
            let mut data = Vec::new();
            std::mem::swap(&mut data, &mut self.data_buffers);
            Ok(data)
        };
        // We don't really care if no one is around to receive it, just let
        // the result go out of scope and get cleaned up
        (self.when_done.take().unwrap())(result);
    }
}

trait DataSink: Send {
    fn deliver_data(&mut self, data: Result<(usize, Bytes)>);
}

impl<F: FnOnce(Result<Vec<Bytes>>) + Send> DataSink for MutableBatch<F> {
    // Called by worker tasks to add data to the MutableBatch
    fn deliver_data(&mut self, data: Result<(usize, Bytes)>) {
        match data {
            Ok(data) => {
                self.data_buffers[data.0] = data.1;
            }
            Err(err) => {
                // This keeps the original error, if present
                self.err.get_or_insert(Box::new(err));
            }
        }
    }
}

struct InnerIoTask {
    reader: Arc<dyn Reader>,
    to_read: Range<u64>,
    when_done: Box<dyn FnOnce(Result<Bytes>) + Send>,
}

impl InnerIoTask {
    async fn run(self) {
        let bytes = self
            .reader
            .get_range(self.to_read.start as usize..self.to_read.end as usize)
            .await;
        (self.when_done)(bytes);
    }
}

// Combines two task receivers into a single stream of tasks
//
// If both receivers have data then items from the priority
// queue will always be taken first.
struct IoQueues {
    regular_queue: mpsc::UnboundedReceiver<InnerIoTask>,
    priority_queue: mpsc::UnboundedReceiver<InnerIoTask>,
}

impl Stream for IoQueues {
    type Item = InnerIoTask;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        // If neither stream is ready then we will poll both of them
        // and they will both register a waker with cx.  So we should get
        // woken when either receiver receives an item
        if let Poll::Ready(task) = self.priority_queue.poll_recv(cx) {
            return Poll::Ready(task);
        }
        self.regular_queue.poll_recv(cx)
    }
}

// Every time a scheduler starts up it launches a task to run the I/O loop.  This loop
// repeats endlessly until the scheduler is destroyed.
async fn run_io_loop(
    tasks: mpsc::UnboundedReceiver<InnerIoTask>,
    priority_tasks: mpsc::UnboundedReceiver<InnerIoTask>,
    io_capacity: u32,
) {
    let io_queues = IoQueues {
        priority_queue: priority_tasks,
        regular_queue: tasks,
    };
    let task_stream = io_queues.map(|task| tokio::spawn(task.run()));
    let mut task_stream = task_stream.buffer_unordered(io_capacity as usize);
    while task_stream.next().await.is_some() {
        // We don't actually do anything with the results here, they are sent
        // via the io tasks's when_done.  Instead we just keep chugging away
        // indefinitely until the tasks receiver returns none (scheduler has
        // been shut down)
    }
}

/// An I/O scheduler which wraps an ObjectStore and throttles the amount of\
/// parallel I/O that can be run.
///
/// TODO: This will also add coalescing
pub struct StoreScheduler {
    object_store: Arc<ObjectStore>,
    io_submitter: mpsc::UnboundedSender<InnerIoTask>,
    priority_sender: Arc<mpsc::UnboundedSender<InnerIoTask>>,
}

impl StoreScheduler {
    /// Create a new scheduler with the given I/O capacity
    ///
    /// # Arguments
    ///
    /// * object_store - the store to wrap
    /// * io_capacity - the maximum number of parallel requests that will be allowed
    pub fn new(object_store: Arc<ObjectStore>, io_capacity: u32) -> Self {
        // TODO: we don't have any backpressure in place if the compute thread falls
        // behind.  The scheduler thread will schedule ALL of the I/O and then the
        // loaded data will eventually pile up.
        //
        // We could bound this channel but that wouldn't help.  If the decode thread
        // was paused then the I/O loop would keep running and reading from this channel.
        //
        // Once the reader is finished we should revisit.  We will probably want to convert
        // from `when_done` futures to delivering data into a queue.  That queue should fill
        // up, causing the I/O loop to pause.
        let (reg_tx, reg_rx) = mpsc::unbounded_channel();
        let (priority_tx, priority_rx) = mpsc::unbounded_channel();
        let scheduler = Self {
            object_store,
            io_submitter: reg_tx,
            priority_sender: Arc::new(priority_tx),
        };
        tokio::task::spawn(async move { run_io_loop(reg_rx, priority_rx, io_capacity).await });
        scheduler
    }

    /// Open a file for reading
    pub async fn open_file(&self, path: &Path) -> Result<FileScheduler> {
        let reader = self.object_store.open(path).await?;
        Ok(FileScheduler {
            reader: reader.into(),
            root: self,
        })
    }

    fn do_submit_request<C: Send + 'static>(
        reader: Arc<dyn Reader>,
        request: BatchRequest<C>,
        tx: oneshot::Sender<Result<LoadedBatch<C>>>,
        priority_submitter: Arc<mpsc::UnboundedSender<InnerIoTask>>,
        this_submitter: &mpsc::UnboundedSender<InnerIoTask>,
    ) {
        let num_iops = request.iops.len() as u32;

        let context = request.context;
        let follow_up = request.follow_up;
        let reader_clone = reader.clone();

        let when_all_io_done = move |bytes| {
            match bytes {
                Ok(bytes) => {
                    if let Some(follow_up) = follow_up {
                        // The current future is on the I/O critical path.  If the follow
                        // up requires any significant CPU work then it will block
                        // the I/O threads and so we run the follow-up in a new task
                        tokio::task::spawn(async move {
                            let next_req = (follow_up)(context, bytes);
                            Self::do_submit_request(
                                reader_clone,
                                next_req,
                                tx,
                                priority_submitter.clone(),
                                &priority_submitter,
                            );
                        });
                    } else {
                        let _ = tx.send(Ok(LoadedBatch {
                            data_buffers: bytes,
                            context,
                        }));
                    }
                }
                Err(err) => {
                    let _ = tx.send(Err(err));
                }
            };
        };

        let dest = Arc::new(Mutex::new(Box::new(MutableBatch::new(
            when_all_io_done,
            num_iops,
        ))));

        for (task_idx, iop) in request.iops.into_iter().enumerate() {
            let dest = dest.clone();
            let task = InnerIoTask {
                reader: reader.clone(),
                to_read: iop,
                when_done: Box::new(move |bytes| {
                    let mut dest = dest.lock().unwrap();
                    dest.deliver_data(bytes.map(|bytes| (task_idx, bytes)));
                }),
            };
            this_submitter
                .send(task)
                .expect("I/O scheduler thread panic'd");
        }
    }

    fn submit_request<C: Send + 'static>(
        &self,
        reader: Arc<dyn Reader>,
        request: BatchRequest<C>,
    ) -> impl Future<Output = Result<LoadedBatch<C>>> + Send {
        let (tx, rx) = oneshot::channel::<Result<LoadedBatch<C>>>();

        Self::do_submit_request(
            reader,
            request,
            tx,
            self.priority_sender.clone(),
            &self.io_submitter,
        );

        // Right now, it isn't possible for I/O to be cancelled so a cancel error should
        // not occur
        rx.map(|wrapped_err| wrapped_err.unwrap())
    }
}

/// A throttled file reader
pub struct FileScheduler<'a> {
    reader: Arc<dyn Reader>,
    root: &'a StoreScheduler,
}

impl<'a> FileScheduler<'a> {
    /// Submit a batch of I/O requests to the reader
    ///
    /// The requests will be queued in a FIFO manner and, when all requests
    /// have been fulfilled, the returned future will be completed.
    pub fn submit_request<C: Send + 'static>(
        &self,
        request: BatchRequest<C>,
    ) -> impl Future<Output = Result<LoadedBatch<C>>> + Send {
        self.root.submit_request(self.reader.clone(), request)
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::VecDeque, mem::size_of};

    use arrow_array::UInt32Array;
    use arrow_buffer::ScalarBuffer;
    use byteorder::{LittleEndian, WriteBytesExt};
    use bytes::BufMut;
    use object_store::path::Path;
    use rand::RngCore;
    use tempfile::tempdir;

    use super::*;

    #[tokio::test]
    async fn test_full_seq_read() {
        let tmpdir = tempdir().unwrap();
        let tmp_path = tmpdir.path().to_str().unwrap();
        let tmp_path = Path::parse(tmp_path).unwrap();
        let tmp_file = tmp_path.child("foo.file");

        let obj_store = Arc::new(ObjectStore::local());

        // Write 1MiB of data
        const DATA_SIZE: u64 = 1024 * 1024;
        let mut some_data = vec![0; DATA_SIZE as usize];
        rand::thread_rng().fill_bytes(&mut some_data);
        obj_store.put(&tmp_file, &some_data).await.unwrap();

        let scheduler = StoreScheduler::new(obj_store, 16);

        let file_scheduler = scheduler.open_file(&tmp_file).await.unwrap();

        // Read it back 4KiB at a time
        const READ_SIZE: u64 = 4 * 1024;
        let mut reqs = VecDeque::new();
        let mut offset = 0;
        while offset < DATA_SIZE {
            reqs.push_back(
                #[allow(clippy::single_range_in_vec_init)]
                file_scheduler
                    .submit_request(BatchRequest::new_simple(vec![offset..offset + READ_SIZE]))
                    .await
                    .unwrap(),
            );
            offset += READ_SIZE;
        }

        offset = 0;
        // Note: we should get parallel I/O even though we are consuming serially
        while offset < DATA_SIZE {
            let data = reqs.pop_front().unwrap();
            let actual = &data.data_buffers[0];
            let expected = &some_data[offset as usize..(offset + READ_SIZE) as usize];
            assert_eq!(expected, actual);
            offset += READ_SIZE;
        }
    }

    // This simulates the variable sized binary decoder
    struct IndirectReadContext {
        base_offset: u64,
        offsets: Option<Vec<UInt32Array>>,
    }

    fn indirect_read_follow_up(
        mut context: IndirectReadContext,
        data: Vec<Bytes>,
    ) -> BatchRequest<IndirectReadContext> {
        let (iops, offsets): (Vec<_>, Vec<_>) = data
            .into_iter()
            .map(|bytes| {
                let length = bytes.len() / size_of::<u32>();
                debug_assert!(length > 1);
                let offsets = ScalarBuffer::<u32>::new(bytes.into(), 0, length);
                let start = context.base_offset + offsets[0] as u64;
                let end = context.base_offset + offsets[length - 1] as u64;
                let offsets = UInt32Array::new(offsets, None);
                (start..end, offsets)
            })
            .unzip();
        context.offsets = Some(offsets);
        BatchRequest::new_with_context(iops, context)
    }

    #[tokio::test]
    async fn test_full_indirect_read() {
        let tmpdir = tempdir().unwrap();
        let tmp_path = tmpdir.path().to_str().unwrap();
        let tmp_path = Path::parse(tmp_path).unwrap();
        let tmp_file = tmp_path.child("foo.file");

        let obj_store = Arc::new(ObjectStore::local());

        // We will pretend this is binary data and write 32Ki strings consisting
        // of 32 characters each (1MiB of data + (256KiB + 8B) of offsets)
        const STRING_WIDTH: u64 = 32;
        const NUM_STRINGS: u64 = 32 * 1024;
        const DATA_SIZE: u64 = STRING_WIDTH * NUM_STRINGS;
        const OFFSET_SIZE: u64 = size_of::<u32>() as u64 * (NUM_STRINGS + 1);
        let mut some_data = Vec::with_capacity((DATA_SIZE + OFFSET_SIZE) as usize);
        // Initialize the offsets section with offsets
        some_data.write_u32::<LittleEndian>(0).unwrap();
        for idx in 0..NUM_STRINGS {
            some_data
                .write_u32::<LittleEndian>((idx as u32 + 1) * 32)
                .unwrap();
        }
        // Initialize the data section with random data
        some_data.put_bytes(0, DATA_SIZE as usize);
        rand::thread_rng().fill_bytes(&mut some_data[OFFSET_SIZE as usize..]);
        obj_store.put(&tmp_file, &some_data).await.unwrap();

        let scheduler = StoreScheduler::new(obj_store, 16);

        let file_scheduler = scheduler.open_file(&tmp_file).await.unwrap();

        // Read it back in one big read
        let indirect_context = IndirectReadContext {
            base_offset: OFFSET_SIZE,
            offsets: None,
        };
        #[allow(clippy::single_range_in_vec_init)]
        let req = BatchRequest::new_with_follow_up(
            vec![0..OFFSET_SIZE],
            indirect_context,
            indirect_read_follow_up,
        );
        let data = file_scheduler.submit_request(req).await.unwrap();

        assert_eq!(data.data_buffers.len(), 1);
        assert_eq!(data.data_buffers[0], some_data[OFFSET_SIZE as usize..]);

        let offsets = data.context.offsets.unwrap().into_iter().next().unwrap();
        assert!(offsets
            .values()
            .iter()
            .enumerate()
            .all(|(idx, len)| *len == (idx as u32 * STRING_WIDTH as u32)));

        // // Read it back in batches
        let mut reqs = VecDeque::new();
        let mut offset = 0;
        const BATCH_SIZE: u64 = 1024 * size_of::<u32>() as u64;
        while offset < OFFSET_SIZE {
            let indirect_context = IndirectReadContext {
                base_offset: OFFSET_SIZE,
                offsets: None,
            };
            #[allow(clippy::single_range_in_vec_init)]
            let req = BatchRequest::new_with_follow_up(
                vec![offset..offset + BATCH_SIZE + size_of::<u32>() as u64],
                indirect_context,
                indirect_read_follow_up,
            );
            reqs.push_back(file_scheduler.submit_request(req));
            offset += BATCH_SIZE;
        }

        let mut data_offset = OFFSET_SIZE;
        const DATA_BATCH_SIZE: u64 = 1024 * STRING_WIDTH;
        // Note: we should get parallel I/O even though we are consuming serially
        while data_offset < DATA_SIZE {
            let data = reqs.pop_front().unwrap().await.unwrap();
            let actual = &data.data_buffers[0];
            let expected =
                &some_data[data_offset as usize..(data_offset + DATA_BATCH_SIZE) as usize];
            assert_eq!(expected, actual);
            data_offset += DATA_BATCH_SIZE;
        }
    }
}
