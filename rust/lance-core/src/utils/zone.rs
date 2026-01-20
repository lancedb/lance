// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Zone-related utilities for Lance data structures

use crate::Result;
use arrow_array::ArrayRef;

/// Zone bound within a fragment
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
    /// Physical row count in the zone (includes deleted rows)
    ///
    /// Calculated as (last_row_offset - first_row_offset + 1)
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
    /// Implementations should automatically reset internal state after emitting
    /// statistics, preparing for the next zone.
    fn finish_zone(&mut self, bound: ZoneBound) -> Result<Self::ZoneStatistics>;
}

/// Builds zones from batches during file writing.
///
/// `FileZoneBuilder` manages zone boundaries and statistics collection for file-level
/// operations. It processes data synchronously in batches without requiring row addresses,
/// making it ideal for writing new data files.
///
pub struct FileZoneBuilder<P: ZoneProcessor> {
    processor: P,
    zone_size: u64,
    current_zone_rows: u64,
    zone_start: u64,
    zones: Vec<P::ZoneStatistics>,
}

impl<P: ZoneProcessor> FileZoneBuilder<P> {
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
    /// when the zone reaches capacity. If a chunk exceeds the zone size, it is split
    /// across multiple zones. The underlying processor's `process_chunk` is called
    /// for statistics computation.
    pub fn process_chunk(&mut self, array: &ArrayRef) -> Result<()> {
        let total_rows = array.len() as u64;
        let mut offset = 0usize;

        while offset < total_rows as usize {
            // Calculate how many rows we can add to the current zone
            let remaining_capacity = self.zone_size - self.current_zone_rows;
            let rows_to_process = (total_rows as usize - offset).min(remaining_capacity as usize);

            // Process the slice
            let slice = array.slice(offset, rows_to_process);
            self.processor.process_chunk(&slice)?;
            self.current_zone_rows += rows_to_process as u64;
            offset += rows_to_process;

            // If zone is full, flush it and start a new one
            if self.current_zone_rows >= self.zone_size {
                self.flush_zone()?;
            }
        }

        Ok(())
    }

    /// Flushes the current zone if it contains any data.
    ///
    /// Creates a `ZoneBound` with the current zone's position and length,
    /// calls the processor's `finish_zone` to compute final statistics
    fn flush_zone(&mut self) -> Result<()> {
        if self.current_zone_rows > 0 {
            let bound = ZoneBound {
                fragment_id: 0, // Placeholder; actual fragment ID assigned during commit
                start: self.zone_start,
                length: self.current_zone_rows as usize,
            };
            let stats = self.processor.finish_zone(bound)?;
            self.zones.push(stats);

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, Int32Array};
    use std::sync::Arc;

    #[derive(Debug, Clone, PartialEq)]
    struct MockStats {
        sum: i32,
        bound: ZoneBound,
    }

    #[derive(Debug)]
    struct MockProcessor {
        current_sum: i32,
    }

    impl MockProcessor {
        fn new() -> Self {
            Self { current_sum: 0 }
        }
    }

    impl ZoneProcessor for MockProcessor {
        type ZoneStatistics = MockStats;

        fn process_chunk(&mut self, values: &ArrayRef) -> Result<()> {
            let arr = values.as_any().downcast_ref::<Int32Array>().unwrap();
            self.current_sum += arr.iter().map(|v| v.unwrap_or(0)).sum::<i32>();
            Ok(())
        }

        fn finish_zone(&mut self, bound: ZoneBound) -> Result<Self::ZoneStatistics> {
            let stats = MockStats {
                sum: self.current_sum,
                bound,
            };
            // Auto-reset for next zone
            self.current_sum = 0;
            Ok(stats)
        }
    }

    fn array_from_vec(values: Vec<i32>) -> ArrayRef {
        Arc::new(Int32Array::from(values))
    }

    #[test]
    fn test_exact_zone_size() {
        // Data that exactly fills one zone
        let processor = MockProcessor::new();
        let mut builder = FileZoneBuilder::new(processor, 4).unwrap();

        let arr = array_from_vec(vec![1, 2, 3, 4]);
        builder.process_chunk(&arr).unwrap();

        let zones = builder.finalize().unwrap();
        assert_eq!(zones.len(), 1);
        assert_eq!(zones[0].sum, 10); // 1+2+3+4
        assert_eq!(zones[0].bound.start, 0);
        assert_eq!(zones[0].bound.length, 4);
    }

    #[test]
    fn test_multiple_full_zones() {
        // Data that fills multiple zones exactly
        let processor = MockProcessor::new();
        let mut builder = FileZoneBuilder::new(processor, 3).unwrap();

        // First zone: 3 rows
        builder
            .process_chunk(&array_from_vec(vec![1, 2, 3]))
            .unwrap();

        // Second zone: 3 rows
        builder
            .process_chunk(&array_from_vec(vec![4, 5, 6]))
            .unwrap();

        // Third zone: 3 rows
        builder
            .process_chunk(&array_from_vec(vec![7, 8, 9]))
            .unwrap();

        let zones = builder.finalize().unwrap();
        assert_eq!(zones.len(), 3);
        assert_eq!(zones[0].sum, 6); // 1+2+3
        assert_eq!(zones[1].sum, 15); // 4+5+6
        assert_eq!(zones[2].sum, 24); // 7+8+9
        assert_eq!(zones[0].bound.start, 0);
        assert_eq!(zones[1].bound.start, 3);
        assert_eq!(zones[2].bound.start, 6);
    }

    #[test]
    fn test_partial_final_zone() {
        // Data that doesn't fill the last zone completely
        let processor = MockProcessor::new();
        let mut builder = FileZoneBuilder::new(processor, 4).unwrap();

        // First zone: exactly 4 rows
        builder
            .process_chunk(&array_from_vec(vec![1, 2, 3, 4]))
            .unwrap();

        // Second zone: only 2 rows (partial)
        builder.process_chunk(&array_from_vec(vec![5, 6])).unwrap();

        let zones = builder.finalize().unwrap();
        assert_eq!(zones.len(), 2);
        assert_eq!(zones[0].sum, 10); // 1+2+3+4
        assert_eq!(zones[1].sum, 11); // 5+6
        assert_eq!(zones[0].bound.start, 0);
        assert_eq!(zones[0].bound.length, 4);
        assert_eq!(zones[1].bound.start, 4);
        assert_eq!(zones[1].bound.length, 2);
    }

    #[test]
    fn test_just_under_zone_size() {
        // Data that is just one row short of zone size
        let processor = MockProcessor::new();
        let mut builder = FileZoneBuilder::new(processor, 5).unwrap();

        builder
            .process_chunk(&array_from_vec(vec![1, 2, 3, 4]))
            .unwrap();

        let zones = builder.finalize().unwrap();
        assert_eq!(zones.len(), 1);
        assert_eq!(zones[0].sum, 10); // 1+2+3+4
        assert_eq!(zones[0].bound.length, 4);
    }

    #[test]
    fn test_just_over_zone_size() {
        // Data that exceeds zone size by a few rows
        // Chunk should be split across multiple zones
        let processor = MockProcessor::new();
        let mut builder = FileZoneBuilder::new(processor, 4).unwrap();

        // 6 rows in one chunk: should create two zones [1,2,3,4] and [5,6]
        builder
            .process_chunk(&array_from_vec(vec![1, 2, 3, 4, 5, 6]))
            .unwrap();

        let zones = builder.finalize().unwrap();
        assert_eq!(zones.len(), 2);
        assert_eq!(zones[0].sum, 10); // 1+2+3+4
        assert_eq!(zones[0].bound.length, 4);
        assert_eq!(zones[1].sum, 11); // 5+6
        assert_eq!(zones[1].bound.start, 4);
        assert_eq!(zones[1].bound.length, 2);
    }

    #[test]
    fn test_multiple_chunks_exceeding_zone() {
        // Multiple small chunks that together exceed zone size
        let processor = MockProcessor::new();
        let mut builder = FileZoneBuilder::new(processor, 5).unwrap();

        // Chunk 1: 2 rows
        builder.process_chunk(&array_from_vec(vec![1, 2])).unwrap();

        // Chunk 2: 2 rows (total: 4, still under)
        builder.process_chunk(&array_from_vec(vec![3, 4])).unwrap();

        // Chunk 3: 2 rows (total: 6, exceeds zone size)
        builder.process_chunk(&array_from_vec(vec![5, 6])).unwrap();

        let zones = builder.finalize().unwrap();
        assert_eq!(zones.len(), 2);
        assert_eq!(zones[0].sum, 15); // 1+2+3+4+5
        assert_eq!(zones[0].bound.length, 5);
        assert_eq!(zones[1].sum, 6); // Just row 6
        assert_eq!(zones[1].bound.start, 5);
        assert_eq!(zones[1].bound.length, 1);
    }

    #[test]
    fn test_zone_size_one() {
        // With zone size = 1, each row triggers a flush
        let processor = MockProcessor::new();
        let mut builder = FileZoneBuilder::new(processor, 1).unwrap();

        // Process one row at a time
        builder.process_chunk(&array_from_vec(vec![10])).unwrap();
        builder.process_chunk(&array_from_vec(vec![20])).unwrap();
        builder.process_chunk(&array_from_vec(vec![30])).unwrap();

        let zones = builder.finalize().unwrap();
        assert_eq!(zones.len(), 3);
        assert_eq!(zones[0].sum, 10);
        assert_eq!(zones[1].sum, 20);
        assert_eq!(zones[2].sum, 30);
        assert_eq!(zones[0].bound.start, 0);
        assert_eq!(zones[1].bound.start, 1);
        assert_eq!(zones[2].bound.start, 2);
    }

    #[test]
    fn test_large_zone_size() {
        // Zone size larger than total data - all data in one zone
        let processor = MockProcessor::new();
        let mut builder = FileZoneBuilder::new(processor, 100).unwrap();

        builder.process_chunk(&array_from_vec(vec![1; 10])).unwrap();

        let zones = builder.finalize().unwrap();
        assert_eq!(zones.len(), 1);
        assert_eq!(zones[0].sum, 10); // 10 ones
        assert_eq!(zones[0].bound.start, 0);
        assert_eq!(zones[0].bound.length, 10);
    }

    #[test]
    fn test_empty_array() {
        // Empty arrays should be handled gracefully
        let processor = MockProcessor::new();
        let mut builder = FileZoneBuilder::new(processor, 4).unwrap();

        builder.process_chunk(&array_from_vec(vec![])).unwrap();

        // Add some real data
        builder
            .process_chunk(&array_from_vec(vec![1, 2, 3, 4]))
            .unwrap();

        let zones = builder.finalize().unwrap();
        assert_eq!(zones.len(), 1);
        assert_eq!(zones[0].sum, 10);
    }

    #[test]
    fn test_processor_reset_between_zones() {
        // Verify processor resets correctly between zones
        let processor = MockProcessor::new();
        let mut builder = FileZoneBuilder::new(processor, 3).unwrap();

        // First zone
        builder
            .process_chunk(&array_from_vec(vec![1, 2, 3]))
            .unwrap();

        // Second zone - processor should have reset, so sum starts from 0
        builder
            .process_chunk(&array_from_vec(vec![4, 5, 6]))
            .unwrap();

        let zones = builder.finalize().unwrap();
        assert_eq!(zones.len(), 2);
        assert_eq!(zones[0].sum, 6);
        assert_eq!(zones[1].sum, 15); // 4+5+6, not 6+15=21
    }

    #[test]
    fn test_zone_boundaries_sequential() {
        // Verify zone start positions are sequential
        // Process in chunks that don't exceed zone size
        let processor = MockProcessor::new();
        let mut builder = FileZoneBuilder::new(processor, 3).unwrap();

        // Process in chunks of 3 (exactly zone size)
        builder
            .process_chunk(&array_from_vec(vec![1, 2, 3]))
            .unwrap();

        builder
            .process_chunk(&array_from_vec(vec![4, 5, 6]))
            .unwrap();

        // Last chunk: 2 rows (partial)
        builder.process_chunk(&array_from_vec(vec![7, 8])).unwrap();

        let zones = builder.finalize().unwrap();
        assert_eq!(zones.len(), 3);
        assert_eq!(zones[0].bound.start, 0);
        assert_eq!(zones[1].bound.start, 3);
        assert_eq!(zones[2].bound.start, 6);
        assert_eq!(zones[0].bound.length, 3);
        assert_eq!(zones[1].bound.length, 3);
        assert_eq!(zones[2].bound.length, 2); // Last partial zone
    }

    #[test]
    fn test_rejects_zero_zone_size() {
        let processor = MockProcessor::new();
        let result = FileZoneBuilder::new(processor, 0);
        assert!(result.is_err());
        let err_msg = format!("{}", result.err().unwrap());
        assert!(err_msg.contains("zone size must be greater than zero"));
    }

    #[test]
    fn test_fragment_id_placeholder() {
        // Verify fragment_id is set to 0 (placeholder) for file-level operations
        let processor = MockProcessor::new();
        let mut builder = FileZoneBuilder::new(processor, 3).unwrap();

        builder
            .process_chunk(&array_from_vec(vec![1, 2, 3]))
            .unwrap();

        let zones = builder.finalize().unwrap();
        assert_eq!(zones[0].bound.fragment_id, 0);
    }

    #[test]
    fn test_edge_case_one_row_short() {
        // Zone size = 5, data = 4 rows (exactly one short)
        let processor = MockProcessor::new();
        let mut builder = FileZoneBuilder::new(processor, 5).unwrap();

        builder
            .process_chunk(&array_from_vec(vec![1, 2, 3, 4]))
            .unwrap();

        let zones = builder.finalize().unwrap();
        assert_eq!(zones.len(), 1);
        assert_eq!(zones[0].bound.length, 4);
    }

    #[test]
    fn test_edge_case_one_row_over() {
        // Zone size = 4, data = 5 rows (exactly one over)
        // Should create two zones: [1,2,3,4] and [5]
        let processor = MockProcessor::new();
        let mut builder = FileZoneBuilder::new(processor, 4).unwrap();

        builder
            .process_chunk(&array_from_vec(vec![1, 2, 3, 4, 5]))
            .unwrap();

        let zones = builder.finalize().unwrap();
        assert_eq!(zones.len(), 2);
        assert_eq!(zones[0].sum, 10); // 1+2+3+4
        assert_eq!(zones[0].bound.length, 4);
        assert_eq!(zones[1].sum, 5); // Just row 5
        assert_eq!(zones[1].bound.start, 4);
        assert_eq!(zones[1].bound.length, 1);
    }

    #[test]
    fn test_large_number_of_small_chunks() {
        // Many small chunks that accumulate
        let processor = MockProcessor::new();
        let mut builder = FileZoneBuilder::new(processor, 10).unwrap();

        // Add 20 chunks of 1 row each
        for i in 1..=20 {
            builder.process_chunk(&array_from_vec(vec![i])).unwrap();
        }

        let zones = builder.finalize().unwrap();
        assert_eq!(zones.len(), 2);
        assert_eq!(zones[0].sum, 55); // Sum of 1..=10
        assert_eq!(zones[1].sum, 155); // Sum of 11..=20
        assert_eq!(zones[0].bound.start, 0);
        assert_eq!(zones[1].bound.start, 10);
    }
}
