// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Zone-related utilities for Lance data structures

use crate::Result;
use arrow_array::ArrayRef;

/// Zone bound within a fragment
///
/// This structure represents the boundary of a zone, which is a contiguous
/// range of rows within a fragment. Zones are used for scalar indexing and
/// column statistics.
///
/// # Fragment ID
///
/// The `fragment_id` field is only meaningful when building zones from existing
/// dataset data (e.g., for index building). When writing new files, this is
/// typically set to 0 as a placeholder since the fragment ID is assigned later
/// during commit.
///
/// # Example
///
/// Suppose we have two fragments, each with 4 rows:
/// - Fragment 0: start = 0, length = 4  // covers rows 0, 1, 2, 3
/// - Fragment 1: start = 0, length = 4  // covers rows 0, 1, 2, 3
///
/// After deleting rows 0 and 1 from fragment 0, and rows 1 and 2 from fragment 1:
/// - Fragment 0: start = 2, length = 2  // covers rows 2, 3
/// - Fragment 1: start = 0, length = 4  // covers rows 0, 3 (with gaps)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZoneBound {
    /// Fragment ID containing this zone
    ///
    /// For file-level operations (e.g., `FileZoneBuilder`), this is typically 0
    /// since the fragment ID is assigned during commit, not during file writing.
    pub fragment_id: u64,
    /// Start row offset within the fragment (local offset)
    ///
    /// To get the actual first row address, use `(fragment_id << 32) | start`.
    pub start: u64,
    /// Span of row offsets between the first and last row in the zone
    ///
    /// Calculated as (last_row_offset - first_row_offset + 1). This is not
    /// the count of physical rows, since deletions may create gaps within
    /// the span.
    pub length: usize,
}

/// Trait for processing data in zones and computing zone-level statistics.
///
/// This trait provides a common interface for zone-based processing used in
/// both scalar indexing (ZoneMap) and file-level column statistics.
///
/// Implementors accumulate statistics as chunks of data are processed, then
/// emit final statistics when a zone is complete.
pub trait ZoneProcessor {
    /// The type of statistics produced for each zone
    type ZoneStatistics;

    /// Process a slice of values that belongs to the current zone.
    ///
    /// This method is called repeatedly with chunks of data. Implementations
    /// should accumulate statistics incrementally.
    fn process_chunk(&mut self, values: &ArrayRef) -> Result<()>;

    /// Emit statistics when the zone is full or the fragment changes.
    ///
    /// The provided `bound` describes the row range covered by this zone.
    /// After calling this method, the processor should be ready to start
    /// accumulating statistics for the next zone (via `reset()`).
    fn finish_zone(&mut self, bound: ZoneBound) -> Result<Self::ZoneStatistics>;

    /// Reset state so the processor can handle the next zone.
    ///
    /// This is called after `finish_zone()` to prepare for processing
    /// the next zone's data.
    fn reset(&mut self) -> Result<()>;
}

/// Builds zones from batches during file writing.
///
/// `FileZoneBuilder` manages zone boundaries and statistics collection for file-level
/// operations. It processes data synchronously in batches without requiring row addresses,
/// making it ideal for writing new data files.
///
/// This builder handles the mechanics of zone management (tracking row counts, flushing
/// zones when full) while delegating statistics computation to a `ZoneProcessor` implementation.
///
/// # Use Cases
///
/// - Writing Lance data files with column statistics
/// - In-memory zone processing for fresh data
/// - Any synchronous, batch-based zone building
///
/// # Contrast with `IndexZoneTrainer`
///
/// For building zones from existing data with row addresses across multiple fragments,
/// use `IndexZoneTrainer` in `lance-index` instead.
///
/// # Example
///
/// ```ignore
/// use lance_core::utils::zone::{FileZoneBuilder, ZoneProcessor};
///
/// let processor = MyZoneProcessor::new(data_type)?;
/// let mut builder = FileZoneBuilder::new(processor, 1_000_000)?;
///
/// for batch in batches {
///     for field in batch.columns() {
///         builder.process_chunk(field)?;
///     }
/// }
///
/// let all_zones = builder.finalize()?;
/// ```
pub struct FileZoneBuilder<P: ZoneProcessor> {
    processor: P,
    zone_size: u64,
    current_zone_rows: u64,
    zone_start: u64,
    zones: Vec<P::ZoneStatistics>,
}

impl<P: ZoneProcessor> FileZoneBuilder<P> {
    /// Creates a new file zone builder.
    ///
    /// # Arguments
    ///
    /// * `processor` - The zone processor that computes statistics
    /// * `zone_size` - Maximum number of rows per zone (e.g., 1,000,000)
    ///
    /// # Errors
    ///
    /// Returns an error if `zone_size` is 0.
    pub fn new(processor: P, zone_size: u64) -> Result<Self> {
        if zone_size == 0 {
            return Err(crate::Error::invalid_input(
                "zone size must be greater than zero",
                snafu::location!(),
            ));
        }
        Ok(Self {
            processor,
            zone_size,
            current_zone_rows: 0,
            zone_start: 0,
            zones: Vec::new(),
        })
    }

    /// Processes a chunk of data, automatically flushing zones when full.
    ///
    /// This method accumulates data into the current zone and automatically flushes
    /// when the zone reaches capacity. The underlying processor's `process_chunk`
    /// is called for statistics computation.
    ///
    /// # Arguments
    ///
    /// * `array` - The array of values to process
    pub fn process_chunk(&mut self, array: &ArrayRef) -> Result<()> {
        let num_rows = array.len() as u64;
        self.processor.process_chunk(array)?;
        self.current_zone_rows += num_rows;

        // If zone is full, finalize it and start a new one
        if self.current_zone_rows >= self.zone_size {
            self.flush_zone()?;
        }

        Ok(())
    }

    /// Flushes the current zone if it contains any data.
    ///
    /// Creates a `ZoneBound` with the current zone's position and length,
    /// calls the processor's `finish_zone` to compute final statistics,
    /// and resets state for the next zone.
    fn flush_zone(&mut self) -> Result<()> {
        if self.current_zone_rows > 0 {
            let bound = ZoneBound {
                fragment_id: 0, // Placeholder; actual fragment ID assigned during commit
                start: self.zone_start,
                length: self.current_zone_rows as usize,
            };
            let stats = self.processor.finish_zone(bound)?;
            self.zones.push(stats);

            // Reset for next zone
            self.processor.reset()?;
            self.zone_start += self.current_zone_rows;
            self.current_zone_rows = 0;
        }
        Ok(())
    }

    /// Finalizes processing and returns all collected zone statistics.
    ///
    /// Flushes any remaining partial zone and consumes the builder,
    /// returning ownership of all zone statistics collected during processing.
    pub fn finalize(mut self) -> Result<Vec<P::ZoneStatistics>> {
        self.flush_zone()?;
        Ok(self.zones)
    }

    /// Returns a reference to the collected zone statistics so far.
    ///
    /// Note: This does not include the current partial zone being accumulated.
    pub fn zones(&self) -> &[P::ZoneStatistics] {
        &self.zones
    }
}
