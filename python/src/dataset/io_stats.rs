// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! IO statistics tracking for dataset operations

use pyo3::{pyclass, pymethods};

/// IO statistics for dataset operations
///
/// This tracks the number of IO operations and bytes transferred for read and write
/// operations performed on the dataset's object store.
///
/// Note: Calling `io_stats()` returns the statistics accumulated since the last call
/// and resets the internal counters (incremental stats pattern).
#[pyclass(name = "IOStats", module = "_lib", get_all)]
#[derive(Clone, Debug)]
pub struct IoStats {
    /// Number of read IO operations performed
    pub read_iops: u64,
    /// Total bytes read from storage
    pub read_bytes: u64,
    /// Number of write IO operations performed
    pub write_iops: u64,
    /// Total bytes written to storage
    pub write_bytes: u64,
    /// Number of disjoint periods where at least one IO is in-flight
    ///
    /// This metric helps understand IO parallelism. A lower number indicates
    /// more parallel IO operations.
    pub num_hops: u64,
}

#[pymethods]
impl IoStats {
    fn __repr__(&self) -> String {
        format!(
            "IOStats(read_iops={}, read_bytes={}, write_iops={}, write_bytes={}, num_hops={})",
            self.read_iops, self.read_bytes, self.write_iops, self.write_bytes, self.num_hops
        )
    }
}

impl IoStats {
    /// Convert from Lance's internal IoStats type
    pub fn from_lance(stats: lance_io::utils::tracking_store::IoStats) -> Self {
        Self {
            read_iops: stats.read_iops,
            read_bytes: stats.read_bytes,
            write_iops: stats.write_iops,
            write_bytes: stats.write_bytes,
            num_hops: stats.num_hops,
        }
    }
}
