// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The lite scheduler's contribution to the global IO counters.
//!
//! Standalone process: `iops_counter()` and `bytes_read_counter()` are
//! process-wide, so sharing a process with other reads would make the exact
//! counts here somebody else's business too.

use std::sync::Arc;

use lance_core::utils::tempfile::TempStdDir;
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::traits::Reader;
use lance_io::{bytes_read_counter, iops_counter};
use object_store::path::Path;

#[tokio::test]
async fn lite_scheduler_publishes_the_io_counters() {
    let tmp = TempStdDir::default();
    let file_path = tmp.join("data.bin");
    std::fs::write(&file_path, vec![7u8; 1024]).unwrap();

    let store = Arc::new(ObjectStore::local());
    let reader: Arc<dyn Reader> = Arc::from(
        store
            .open(&Path::from_absolute_path(&file_path).unwrap())
            .await
            .unwrap(),
    );

    let scheduler = ScanScheduler::new(
        store,
        SchedulerConfig::default_for_testing().with_lite_scheduler(),
    );

    let iops_before = iops_counter();
    let bytes_before = bytes_read_counter();

    let read = scheduler
        .submit_request(reader.clone(), vec![0..100, 100..300], 0, false)
        .await
        .unwrap();
    assert_eq!(read.iter().map(|bytes| bytes.len()).sum::<usize>(), 300);

    // An empty range reaches the queue, and the standard path counts nothing
    // for one.
    scheduler
        .submit_request(reader, vec![300..300], 0, false)
        .await
        .unwrap();

    assert_eq!(iops_counter() - iops_before, 2);
    assert_eq!(bytes_read_counter() - bytes_before, 300);
}
