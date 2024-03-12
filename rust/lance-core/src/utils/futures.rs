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

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    task::Waker,
};

use futures::{stream::BoxStream, Stream, StreamExt};
use tokio::sync::Semaphore;
use tokio_util::sync::PollSemaphore;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Side {
    Left,
    Right,
}

/// A potentially unbounded capacity
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Capacity {
    Bounded(u32),
    Unbounded,
}

struct InnerState<'a, T> {
    inner: Option<BoxStream<'a, T>>,
    buffer: VecDeque<T>,
    polling: Option<Side>,
    waker: Option<Waker>,
    exhausted: bool,
    left_buffered: u32,
    right_buffered: u32,
    available_buffer: Option<PollSemaphore>,
}

/// The stream returned by [`share`].
pub struct SharedStream<'a, T: Clone> {
    state: Arc<Mutex<InnerState<'a, T>>>,
    side: Side,
}

impl<'a, T: Clone> SharedStream<'a, T> {
    pub fn new(inner: BoxStream<'a, T>, capacity: Capacity) -> (Self, Self) {
        let available_buffer = match capacity {
            Capacity::Unbounded => None,
            Capacity::Bounded(capacity) => Some(PollSemaphore::new(Arc::new(Semaphore::new(
                capacity as usize,
            )))),
        };
        let state = InnerState {
            inner: Some(inner),
            buffer: VecDeque::new(),
            polling: None,
            waker: None,
            exhausted: false,
            left_buffered: 0,
            right_buffered: 0,
            available_buffer,
        };

        let state = Arc::new(Mutex::new(state));

        let left = Self {
            state: state.clone(),
            side: Side::Left,
        };
        let right = Self {
            state,
            side: Side::Right,
        };
        (left, right)
    }
}

impl<'a, T: Clone> Stream for SharedStream<'a, T> {
    type Item = T;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let mut inner_state = self.state.lock().unwrap();
        let can_take_buffered = match self.side {
            Side::Left => inner_state.left_buffered > 0,
            Side::Right => inner_state.right_buffered > 0,
        };
        if can_take_buffered {
            // Easy case, there is an item in the buffer.  Grab it, decrement the count, and return it.
            let item = inner_state.buffer.pop_front();
            match self.side {
                Side::Left => {
                    inner_state.left_buffered -= 1;
                }
                Side::Right => {
                    inner_state.right_buffered -= 1;
                }
            }
            if let Some(available_buffer) = inner_state.available_buffer.as_mut() {
                available_buffer.add_permits(1);
            }
            std::task::Poll::Ready(item)
        } else {
            if inner_state.exhausted {
                return std::task::Poll::Ready(None);
            }
            // No buffered items, if we have room in the buffer, then try and poll for one
            let permit = if let Some(available_buffer) = inner_state.available_buffer.as_mut() {
                match available_buffer.poll_acquire(cx) {
                    // Can return None if the semaphore is closed but we never close the semaphore
                    // so its safe to unwrap here
                    std::task::Poll::Ready(permit) => Some(permit.unwrap()),
                    std::task::Poll::Pending => {
                        return std::task::Poll::Pending;
                    }
                }
            } else {
                None
            };
            if let Some(polling_side) = inner_state.polling.as_ref() {
                if *polling_side != self.side {
                    // Another task is already polling the inner stream, so we don't need to do anything
                    debug_assert!(inner_state.waker.is_none());
                    inner_state.waker = Some(cx.waker().clone());
                    return std::task::Poll::Pending;
                }
            }
            inner_state.polling = Some(self.side);
            // Release the mutex here as polling the inner stream is potentially expensive
            let mut to_poll = inner_state
                .inner
                .take()
                .expect("Other half of shared stream panic'd while polling inner stream");
            drop(inner_state);
            let res = to_poll.poll_next_unpin(cx);
            let mut inner_state = self.state.lock().unwrap();

            let mut should_wake = true;
            match &res {
                std::task::Poll::Ready(None) => {
                    inner_state.exhausted = true;
                    inner_state.polling = None;
                }
                std::task::Poll::Ready(Some(item)) => {
                    // We got an item, forget the permit to mark that we can take one fewer items
                    if let Some(permit) = permit {
                        permit.forget();
                    }
                    inner_state.polling = None;
                    // Let the other side know an item is available
                    match self.side {
                        Side::Left => {
                            inner_state.right_buffered += 1;
                        }
                        Side::Right => {
                            inner_state.left_buffered += 1;
                        }
                    };
                    inner_state.buffer.push_back(item.clone());
                }
                std::task::Poll::Pending => {
                    should_wake = false;
                }
            };

            inner_state.inner = Some(to_poll);

            // If the other side was waiting for us to poll, wake them up, but only after we release the mutex
            let to_wake = if should_wake {
                inner_state.waker.take()
            } else {
                // If the inner stream is pending then the inner stream will wake us up and we will wake the
                // other side up then.
                None
            };
            drop(inner_state);
            if let Some(waker) = to_wake {
                waker.wake();
            }
            res
        }
    }
}

pub trait SharedStreamExt<'a>: Stream + Send
where
    Self::Item: Clone,
{
    /// Split a stream into two shared streams
    ///
    /// Each shared stream will return the full set of items from the underlying stream.
    /// This works by buffering the items from the underlying stream and then replaying
    /// them to the other side.
    ///
    /// The capacity parameter controls how many items can be buffered at once.  Be careful
    /// with the capacity parameter as it can lead to deadlock if the two streams are not
    /// polled evenly.
    ///
    /// If the capacity is unbounded then the stream could potentially buffer the entire
    /// input stream in memory.
    fn share(
        self,
        capacity: Capacity,
    ) -> (SharedStream<'a, Self::Item>, SharedStream<'a, Self::Item>);
}

impl<'a, T: Clone> SharedStreamExt<'a> for BoxStream<'a, T> {
    fn share(self, capacity: Capacity) -> (SharedStream<'a, T>, SharedStream<'a, T>) {
        SharedStream::new(self, capacity)
    }
}

#[cfg(test)]
mod tests {

    use futures::{FutureExt, StreamExt};
    use tokio_stream::wrappers::ReceiverStream;

    use crate::utils::futures::{Capacity, SharedStreamExt};

    fn is_pending(fut: &mut (impl std::future::Future + Unpin)) -> bool {
        let noop_waker = futures::task::noop_waker();
        let mut context = std::task::Context::from_waker(&noop_waker);
        fut.poll_unpin(&mut context).is_pending()
    }

    #[tokio::test]
    async fn test_shared_stream() {
        let (tx, rx) = tokio::sync::mpsc::channel::<u32>(10);
        let inner_stream = ReceiverStream::new(rx);

        // Feed in a few items
        for i in 0..3 {
            tx.send(i).await.unwrap();
        }

        let (mut left, mut right) = inner_stream.boxed().share(Capacity::Bounded(2));

        // We should be able to immediately poll 2 items
        assert_eq!(left.next().await.unwrap(), 0);
        assert_eq!(left.next().await.unwrap(), 1);

        // Polling again should block because the right side has fallen behind
        let mut left_fut = left.next();

        assert!(is_pending(&mut left_fut));

        // Polling the right side should yield the first cached item and unblock the left
        assert_eq!(right.next().await.unwrap(), 0);
        assert_eq!(left_fut.await.unwrap(), 2);

        // Drain the rest of the stream from the right
        assert_eq!(right.next().await.unwrap(), 1);
        assert_eq!(right.next().await.unwrap(), 2);

        // The channel isn't closed yet so we should get pending on both sides
        let mut right_fut = right.next();
        let mut left_fut = left.next();
        assert!(is_pending(&mut right_fut));
        assert!(is_pending(&mut left_fut));

        // Send one more item
        tx.send(3).await.unwrap();

        // Should be received by both
        assert_eq!(right_fut.await.unwrap(), 3);
        assert_eq!(left_fut.await.unwrap(), 3);

        drop(tx);

        // Now we should be able to poll the end from either side
        assert_eq!(left.next().await, None);
        assert_eq!(right.next().await, None);

        // We should be self-fused
        assert_eq!(left.next().await, None);
        assert_eq!(right.next().await, None);
    }

    #[tokio::test]
    async fn test_unbounded_shared_stream() {
        let (tx, rx) = tokio::sync::mpsc::channel::<u32>(10);
        let inner_stream = ReceiverStream::new(rx);

        // Feed in a few items
        for i in 0..10 {
            tx.send(i).await.unwrap();
        }
        drop(tx);

        let (mut left, mut right) = inner_stream.boxed().share(Capacity::Unbounded);

        // We should be able to completely drain one side
        for i in 0..10 {
            assert_eq!(left.next().await.unwrap(), i);
        }
        assert_eq!(left.next().await, None);

        // And still drain the other side from the buffer
        for i in 0..10 {
            assert_eq!(right.next().await.unwrap(), i);
        }
        assert_eq!(right.next().await, None);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn stress_shared_stream() {
        for _ in 0..100 {
            let (tx, rx) = tokio::sync::mpsc::channel::<u32>(10);
            let inner_stream = ReceiverStream::new(rx);
            let (mut left, mut right) = inner_stream.boxed().share(Capacity::Bounded(2));

            let left_handle = tokio::spawn(async move {
                let mut counter = 0;
                while let Some(item) = left.next().await {
                    assert_eq!(item, counter);
                    counter += 1;
                }
            });

            let right_handle = tokio::spawn(async move {
                let mut counter = 0;
                while let Some(item) = right.next().await {
                    assert_eq!(item, counter);
                    counter += 1;
                }
            });

            for i in 0..1000 {
                tx.send(i).await.unwrap();
            }
            drop(tx);
            left_handle.await.unwrap();
            right_handle.await.unwrap();
        }
    }
}
