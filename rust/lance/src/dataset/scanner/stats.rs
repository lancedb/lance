// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    str::FromStr,
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};

use arrow_array::RecordBatch;
use lance_core::{Error, Result};
use snafu::{location, Location};

pub struct ScannerStats {
    /// The start of the scan
    ///
    /// This is when the stream is constructed, not when it is first consumed.
    pub start: SystemTime,
    /// The end of the scan
    ///
    /// This is when the last batch is provided to the consumer which may be
    /// well after the I/O has finished (if there is a slow consumer or expensive
    /// decode).
    pub end: SystemTime,
    /// The wall clock duration of the scan
    ///
    /// NOTE: This is not the time that the scanner was actually doing work, and not the amount
    /// of time spent in I/O but simply the time from when the scanner was created to when the
    /// last batch was provided to the consumer.
    ///
    /// As an example, if a consumer is slow to consume the data (e.g. they are writing the data
    /// back out to disk or doing expensive processing) then this will be much larger than the
    /// actual time to read the data.
    pub wall_clock_duration: Duration,
    /// The number of rows output by the scanner
    pub output_rows: u64,
    /// The estimated size of the output in bytes
    ///
    /// "Estimated" is used here because there may be some instances where multiple
    /// batches will share the same underlying buffer (e.g. a dictionary) and so the
    /// actual data size may be less than the reported size.
    ///
    /// Also, this is very different than "input bytes" which may be much smaller since
    /// the input may be compressed or encoded.
    ///
    /// This will always be greater than or equal to the actual size.
    pub estimated_output_bytes: u64,
    /// The plan that was used to generate the scan
    ///
    /// There are some instances where we generate a scan without a plan and some handlers
    /// do not need the plan and so we may not gather it.  In these cases this will be None.
    pub plan: Option<String>,
}

impl ScannerStats {
    /// This is an estimate of the "wall clock throughput" in GiB/s
    ///
    /// Note: this is based both on [`Self::wall_clock_duration`] (see note on that method) and
    /// [`Self::estimated_output_bytes`] (see note on that field).
    ///
    /// It is not safe, for example, to assume that this is the rate at which data was pulled down
    /// from storage.
    pub fn wall_clock_throughput(&self) -> impl ThroughputUnit {
        let duration_secs = self.wall_clock_duration.as_secs_f64();
        if duration_secs == 0.0 {
            return 0.0;
        }
        self.estimated_output_bytes as f64 / duration_secs
    }
}

pub trait ThroughputUnit {
    fn gigabytes_per_second(&self) -> f64;
}

/// Here we assume that the throughput is in B/s
impl ThroughputUnit for f64 {
    fn gigabytes_per_second(&self) -> f64 {
        self / (1024.0 * 1024.0 * 1024.0)
    }
}

pub(super) struct ScannerStatsCollector {
    start: Instant,
    start_time: SystemTime,
    output_rows: u64,
    estimated_output_bytes: u64,
    plan: Option<String>,
    handler: ScanStatisticsHandler,
}

impl ScannerStatsCollector {
    pub fn new(handler: ScanStatisticsHandler, plan: Option<String>) -> Self {
        let start = Instant::now();
        let start_time = SystemTime::now();
        Self {
            start,
            start_time,
            output_rows: 0,
            estimated_output_bytes: 0,
            plan,
            handler,
        }
    }

    pub fn observe_batch(&mut self, batch: &RecordBatch) {
        self.output_rows += batch.num_rows() as u64;
        self.estimated_output_bytes += batch
            .columns()
            .iter()
            .map(|c| c.get_buffer_memory_size() as u64)
            .sum::<u64>();
    }

    pub fn finish(self) -> Result<()> {
        let end = Instant::now();
        let end_time = SystemTime::now();
        let stats = ScannerStats {
            start: self.start_time,
            end: end_time,
            wall_clock_duration: (end - self.start),
            output_rows: self.output_rows,
            estimated_output_bytes: self.estimated_output_bytes,
            plan: self.plan,
        };
        match self.handler {
            ScanStatisticsHandler::DoNotReport => Ok(()),
            ScanStatisticsHandler::LogBrief => {
                log::debug!(
                    "Scan wall time {}s ({} GiB/s), output {} rows, estimated output size {} bytes",
                    stats.wall_clock_throughput().gigabytes_per_second(),
                    stats.wall_clock_duration.as_secs_f64(),
                    stats.output_rows,
                    stats.estimated_output_bytes
                );
                Ok(())
            }
            ScanStatisticsHandler::LogFull => {
                log::debug!(
                    "Scan wall time {}s ({} GiB/s), output {} rows, estimated output size {} bytes, plan: {}",
                    stats.wall_clock_duration.as_secs_f64(),
                    stats.wall_clock_throughput().gigabytes_per_second(),
                    stats.output_rows,
                    stats.estimated_output_bytes,
                    stats.plan.as_deref().unwrap_or("N/A")
                );
                Ok(())
            }
            ScanStatisticsHandler::Custom(handler) => handler(stats),
        }
    }
}

/// Describes how statistics should be handled
#[derive(Clone)]
pub enum ScanStatisticsHandler {
    /// Do not report (and possibly even do not gather) any statistics
    DoNotReport,
    /// Log the scan statistics at the end of the scan
    LogBrief,
    /// Log the scan statistics (and the scan plan) at the end of the scan
    LogFull,
    /// Call a custom function with the statistics at the end of the scan
    Custom(Arc<dyn Fn(ScannerStats) -> Result<()> + Send + Sync>),
}

impl std::fmt::Debug for ScanStatisticsHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DoNotReport => write!(f, "DoNotReport"),
            Self::LogBrief => write!(f, "LogBrief"),
            Self::LogFull => write!(f, "LogFull"),
            Self::Custom(_) => write!(f, "Custom"),
        }
    }
}

impl FromStr for ScanStatisticsHandler {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "brief" => Ok(Self::LogBrief),
            "full" => Ok(Self::LogFull),
            _ => Err(Error::InvalidInput {
                source: format!("invalid value for ScanStatisticsHandler: {}", s).into(),
                location: location!(),
            }),
        }
    }
}

lazy_static::lazy_static! {
    pub(crate) static ref DEFAULT_STATS_HANDLER: ScanStatisticsHandler = match std::env::var("LANCE_SCAN_STATISTICS") {
        Ok(val) => match val.as_str() {
            "brief" => ScanStatisticsHandler::LogBrief,
            "full" => ScanStatisticsHandler::LogFull,
            _ => ScanStatisticsHandler::DoNotReport,
        },
        Err(_) => ScanStatisticsHandler::DoNotReport,
    };
}
