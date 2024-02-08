use std::{ops::Range, sync::Arc};

use arrow_array::{ArrayRef, FixedSizeListArray};
use arrow_schema::Field;
use futures::future::BoxFuture;
use log::trace;
use tokio::sync::mpsc;

use crate::{
    decoder::{DecodeArrayTask, LogicalPageDecoder, LogicalPageScheduler, NextDecodeTask},
    EncodingsIo,
};
use lance_core::Result;

// The Fsl decoder relies on the fact that the inner child decoder will never split a list across two
// pages.  As a result, each child page of X rows can map to an FSL page with X / DIM rows.
//
// This is simpler than the List case where a single List page may have multiple child pages.
//
// Nulls are not supported yet.  This means that an Fsl page has zero buffers of its own.  In the future
// this may need to change as there is no guarantee that an FSL will always have a sentinel (e.g. imagine
// an FSL of [u8; 2] that covers every 2^16 possible values and all combinations of NULL),  This could
// fit within a single page.

// Note: The fixed size list encoding is both a logical and a physical encoding.  This file contains the
// logical encoding which is only used when the inner type is, itself, a logical encoding.  For example,
// a fixed-size-list<struct<...>> would use this encoding.

// TODO: There are no tests for this yet.

#[derive(Debug)]
pub struct FslPageScheduler {
    items_scheduler: Box<dyn LogicalPageScheduler>,
    dimension: u32,
}

impl FslPageScheduler {
    pub fn new(items_scheduler: Box<dyn LogicalPageScheduler>, dimension: u32) -> Self {
        debug_assert_eq!(items_scheduler.num_rows() % dimension, 0);
        Self {
            items_scheduler,
            dimension,
        }
    }
}

impl LogicalPageScheduler for FslPageScheduler {
    fn schedule_range(
        &self,
        range: Range<u32>,
        scheduler: &Arc<dyn EncodingsIo>,
        sink: &mpsc::UnboundedSender<Box<dyn LogicalPageDecoder>>,
    ) -> Result<()> {
        let expanded_range = (range.start * self.dimension)..(range.end * self.dimension);
        trace!("Scheduling range {:?} from items scheduler", expanded_range);
        let (tx, mut rx) = mpsc::unbounded_channel();
        self.items_scheduler
            .schedule_range(expanded_range, scheduler, &tx)?;
        let inner_page_decoder = rx.blocking_recv().unwrap();
        sink.send(Box::new(FslPageDecoder {
            inner: inner_page_decoder,
            dimension: self.dimension,
        }))
        .unwrap();
        Ok(())
    }

    fn num_rows(&self) -> u32 {
        self.items_scheduler.num_rows() / self.dimension
    }
}

struct FslPageDecoder {
    inner: Box<dyn LogicalPageDecoder>,
    dimension: u32,
}

impl LogicalPageDecoder for FslPageDecoder {
    fn wait<'a>(
        &'a mut self,
        num_rows: u32,
        source: &'a mut mpsc::UnboundedReceiver<Box<dyn LogicalPageDecoder>>,
    ) -> BoxFuture<'a, Result<()>> {
        self.inner.wait(num_rows * self.dimension, source)
    }

    fn unawaited(&self) -> u32 {
        self.inner.unawaited() / self.dimension
    }

    fn drain(&mut self, num_rows: u32) -> Result<NextDecodeTask> {
        self.inner
            .drain(num_rows * self.dimension)
            .map(|inner_task| {
                let task = Box::new(FslDecodeTask {
                    inner: inner_task.task,
                    dimension: self.dimension,
                });
                NextDecodeTask {
                    task,
                    num_rows: inner_task.num_rows / self.dimension,
                    has_more: inner_task.has_more,
                }
            })
    }

    fn avail(&self) -> u32 {
        self.inner.avail() / self.dimension
    }
}

struct FslDecodeTask {
    inner: Box<dyn DecodeArrayTask>,
    dimension: u32,
}

impl DecodeArrayTask for FslDecodeTask {
    fn decode(self: Box<Self>) -> Result<ArrayRef> {
        let child_array = self.inner.decode()?;
        Ok(Arc::new(FixedSizeListArray::new(
            Arc::new(Field::new("item", child_array.data_type().clone(), true)),
            self.dimension as i32,
            child_array,
            // TODO: Support nullable FSL
            None,
        )))
    }
}
