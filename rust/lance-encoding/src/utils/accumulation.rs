// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! An accumulation queue accumulates arrays until we have enough data to flush.

use std::sync::{Arc, Weak};

use arrow_array::{Array, ArrayRef, cast::AsArray, make_array};
use arrow_data::ArrayData;
use arrow_schema::DataType;
use lance_arrow::deepcopy::deep_copy_array_sliced;
use lance_arrow::memory::{array_slice_memory_size, array_slice_memory_size_parts};
use log::{debug, trace};

#[derive(Clone, Debug, Eq, PartialEq)]
struct ArrayDataIdentity {
    data_type: DataType,
    len: usize,
    offset: usize,
    buffers: Vec<(usize, usize)>,
    nulls: Option<(usize, usize, usize, usize)>,
    children: Vec<Self>,
}

impl ArrayDataIdentity {
    fn new(data: &ArrayData) -> Self {
        let buffers = data
            .buffers()
            .iter()
            .map(|buffer| (buffer.as_ptr() as usize, buffer.len()))
            .collect();
        let nulls = data.nulls().map(|nulls| {
            let values = nulls.inner();
            (
                values.inner().as_ptr() as usize,
                values.inner().len(),
                values.offset(),
                values.len(),
            )
        });
        let children = data.child_data().iter().map(Self::new).collect();
        Self {
            data_type: data.data_type().clone(),
            len: data.len(),
            offset: data.offset(),
            buffers,
            nulls,
            children,
        }
    }
}

#[derive(Debug)]
struct CachedDictionaryValues {
    source: Weak<dyn Array>,
    identity: ArrayDataIdentity,
    values: ArrayRef,
}

impl CachedDictionaryValues {
    fn matches(
        &mut self,
        source: &ArrayRef,
        identity: &ArrayDataIdentity,
        allow_equivalent_values: bool,
    ) -> bool {
        if self.identity != *identity {
            // Normalizing nullable dictionary values creates fresh buffers for
            // every input slice. Reuse the detached child when those buffers
            // still contain the same logical values.
            if allow_equivalent_values && source.to_data() == self.values.to_data() {
                self.source = Arc::downgrade(source);
                self.identity = identity.clone();
                return true;
            }
            return false;
        }

        if self.source.upgrade().is_some() {
            // The allocation represented by `identity` is still live, so its
            // buffer addresses cannot have been reused for unrelated data.
            return true;
        }

        // A raw address may be reused after the original wrapper and buffers
        // are gone. Validate against the detached logical values before
        // adopting a new wrapper with the same buffer fingerprint.
        if source.to_data() == self.values.to_data() {
            self.source = Arc::downgrade(source);
            true
        } else {
            false
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DictionaryKeySlice {
    allocation: (usize, usize),
    start: usize,
    end: usize,
}

impl DictionaryKeySlice {
    fn new(keys: &dyn Array) -> Option<Self> {
        let data = keys.to_data();
        let buffer = data.buffers().first()?;
        let start = buffer.ptr_offset();
        let end = start.checked_add(buffer.len())?;
        Some(Self {
            allocation: (buffer.data_ptr().as_ptr() as usize, buffer.capacity()),
            start,
            end,
        })
    }

    fn has_unreferenced_values(&self) -> bool {
        self.allocation.1 > 0 && self.end < self.allocation.1
    }
}

#[derive(Clone, Debug)]
struct DictionaryRun {
    values: usize,
    key_allocation: (usize, usize),
    next_key_offset: usize,
    has_oversized_shared_values: bool,
}

impl DictionaryRun {
    fn new(values: usize, keys: &DictionaryKeySlice, has_oversized_shared_values: bool) -> Self {
        Self {
            values,
            key_allocation: keys.allocation,
            next_key_offset: keys.end,
            has_oversized_shared_values,
        }
    }

    fn continues(
        &self,
        values: usize,
        keys: &DictionaryKeySlice,
        has_oversized_shared_values: bool,
    ) -> bool {
        self.values == values
            && ((self.has_oversized_shared_values && has_oversized_shared_values)
                || (self.key_allocation == keys.allocation && self.next_key_offset == keys.start))
    }
}

struct DictionaryArrayInfo {
    values: ArrayRef,
    values_identity: ArrayDataIdentity,
    keys: Option<DictionaryKeySlice>,
    shared_bytes: u64,
    incremental_bytes: u64,
}

impl DictionaryArrayInfo {
    fn new(array: &ArrayRef) -> Option<Self> {
        let dictionary = array.as_any_dictionary_opt()?;
        let values = dictionary.values().clone();
        let size = array_slice_memory_size_parts(array.as_ref());
        Some(Self {
            values_identity: ArrayDataIdentity::new(&values.to_data()),
            keys: DictionaryKeySlice::new(dictionary.keys()),
            values,
            shared_bytes: size.shared as u64,
            incremental_bytes: size.incremental as u64,
        })
    }
}

#[derive(Debug)]
pub struct AccumulationQueue {
    cache_bytes: u64,
    keep_original_array: bool,
    use_shared_dictionary_sizing: bool,
    buffered_arrays: Vec<ArrayRef>,
    dictionary_values: Vec<CachedDictionaryValues>,
    dictionary_run: Option<DictionaryRun>,
    current_bytes: u64,
    current_incremental_bytes: u64,
    // Row number of the first item in buffered_arrays, reset on flush
    row_number: u64,
    // Number of top level rows represented in buffered_arrays, reset on flush
    num_rows: u64,
    // This is only for logging / debugging purposes
    column_index: u32,
}

impl AccumulationQueue {
    pub fn new(cache_bytes: u64, column_index: u32, keep_original_array: bool) -> Self {
        Self {
            cache_bytes,
            buffered_arrays: Vec::new(),
            dictionary_values: Vec::new(),
            dictionary_run: None,
            current_bytes: 0,
            current_incremental_bytes: 0,
            column_index,
            keep_original_array,
            use_shared_dictionary_sizing: true,
            row_number: u64::MAX,
            num_rows: 0,
        }
    }

    /// Retain the historical accounting used by stable writer entry points.
    pub(crate) fn with_legacy_dictionary_sizing(mut self) -> Self {
        self.use_shared_dictionary_sizing = false;
        self
    }

    /// Adds an array to the queue, if there is enough data then the queue is flushed
    /// and returned
    pub fn insert(
        &mut self,
        array: ArrayRef,
        row_number: u64,
        num_rows: u64,
    ) -> Option<(Vec<ArrayRef>, u64, u64)> {
        self.insert_with_additional_bytes(array, row_number, num_rows, 0)
    }

    /// Adds an array while including pending structural bytes in run bounds.
    pub(crate) fn insert_with_additional_bytes(
        &mut self,
        array: ArrayRef,
        row_number: u64,
        num_rows: u64,
        pending_additional_bytes: u64,
    ) -> Option<(Vec<ArrayRef>, u64, u64)> {
        if self.row_number == u64::MAX {
            self.row_number = row_number;
        }
        self.num_rows += num_rows;
        let dictionary = DictionaryArrayInfo::new(&array);
        let cached_values = dictionary.as_ref().and_then(|dictionary| {
            self.find_dictionary_values(&dictionary.values, &dictionary.values_identity)
        });
        let dictionary_values = cached_values.unwrap_or(self.dictionary_values.len());
        let (inserted_bytes, inserted_incremental_bytes) = if self.use_shared_dictionary_sizing {
            dictionary.as_ref().map_or_else(
                || {
                    let bytes = array_slice_memory_size(array.as_ref()) as u64;
                    (bytes, bytes)
                },
                |dictionary| {
                    (
                        dictionary.incremental_bytes
                            + if cached_values.is_some() {
                                0
                            } else {
                                dictionary.shared_bytes
                            },
                        dictionary.incremental_bytes,
                    )
                },
            )
        } else {
            let bytes = array_slice_memory_size(array.as_ref()) as u64;
            (bytes, bytes)
        };
        self.current_bytes = self.current_bytes.saturating_add(inserted_bytes);
        self.current_incremental_bytes = self
            .current_incremental_bytes
            .saturating_add(inserted_incremental_bytes);

        let continues_dictionary_run = dictionary
            .as_ref()
            .and_then(|dictionary| {
                dictionary.keys.as_ref().map(|keys| {
                    self.dictionary_run.as_ref().is_some_and(|run| {
                        run.continues(
                            dictionary_values,
                            keys,
                            dictionary.shared_bytes > self.cache_bytes,
                        )
                    })
                })
            })
            .unwrap_or(false);
        let starts_sliced_dictionary_run = self.buffered_arrays.is_empty()
            && dictionary.as_ref().is_some_and(|dictionary| {
                dictionary
                    .keys
                    .as_ref()
                    .is_some_and(DictionaryKeySlice::has_unreferenced_values)
                    || dictionary.shared_bytes > self.cache_bytes
            });
        let defer_dictionary_flush = self.use_shared_dictionary_sizing
            && (continues_dictionary_run || starts_sliced_dictionary_run)
            && self
                .current_incremental_bytes
                .saturating_add(pending_additional_bytes)
                <= self.cache_bytes;

        if self.current_bytes > self.cache_bytes && !defer_dictionary_flush {
            debug!(
                "Flushing column {} page of size {} bytes (unencoded)",
                self.column_index, self.current_bytes
            );
            // Dictionary slices already in the queue use detached values. Reuse
            // those values for the final slice so concatenation still recognizes
            // one shared dictionary. Other arrays need no copy before a flush.
            let array = match (dictionary.as_ref(), cached_values) {
                (Some(_), Some(index)) if !self.keep_original_array => {
                    self.copy_dictionary_array(&array, index, false)
                }
                _ => array,
            };
            self.buffered_arrays.push(array);
            self.current_bytes = 0;
            self.current_incremental_bytes = 0;
            let row_number = self.row_number;
            self.row_number = u64::MAX;
            let num_rows = self.num_rows;
            self.num_rows = 0;
            Some((self.take_buffered_arrays(), row_number, num_rows))
        } else {
            trace!(
                "Accumulating data for column {}.  Now at {} bytes",
                self.column_index, self.current_bytes
            );
            let cached_values = dictionary
                .as_ref()
                .map(|dictionary| self.cache_dictionary_values(dictionary, cached_values));
            let array = if self.keep_original_array {
                array
            } else if let Some(index) = cached_values {
                self.copy_dictionary_array(&array, index, defer_dictionary_flush)
            } else {
                deep_copy_array_sliced(array.as_ref())
            };
            self.record_dictionary_run(dictionary.as_ref(), cached_values);
            self.buffered_arrays.push(array);
            None
        }
    }

    pub fn flush(&mut self) -> Option<(Vec<ArrayRef>, u64, u64)> {
        if self.buffered_arrays.is_empty() {
            trace!(
                "No final flush since no data at column {}",
                self.column_index
            );
            None
        } else {
            trace!(
                "Final flush of column {} which has {} bytes",
                self.column_index, self.current_bytes
            );
            self.current_bytes = 0;
            self.current_incremental_bytes = 0;
            let row_number = self.row_number;
            self.row_number = u64::MAX;
            let num_rows = self.num_rows;
            self.num_rows = 0;
            Some((self.take_buffered_arrays(), row_number, num_rows))
        }
    }

    /// Estimated bytes retained by arrays that have not been emitted yet.
    pub fn pending_bytes(&self) -> u64 {
        self.current_bytes
    }

    fn find_dictionary_values(
        &mut self,
        values: &ArrayRef,
        identity: &ArrayDataIdentity,
    ) -> Option<usize> {
        let allow_equivalent_values = self.use_shared_dictionary_sizing;
        self.dictionary_values
            .iter_mut()
            .position(|cached| cached.matches(values, identity, allow_equivalent_values))
    }

    fn cache_dictionary_values(
        &mut self,
        dictionary: &DictionaryArrayInfo,
        cached_values: Option<usize>,
    ) -> usize {
        cached_values.unwrap_or_else(|| {
            let values = if self.keep_original_array {
                dictionary.values.clone()
            } else {
                deep_copy_array_sliced(dictionary.values.as_ref())
            };
            let index = self.dictionary_values.len();
            self.dictionary_values.push(CachedDictionaryValues {
                source: Arc::downgrade(&dictionary.values),
                identity: dictionary.values_identity.clone(),
                values,
            });
            index
        })
    }

    fn copy_dictionary_array(
        &self,
        array: &ArrayRef,
        dictionary_values: usize,
        keep_source_keys: bool,
    ) -> ArrayRef {
        let dictionary = array.as_any_dictionary();
        let keys = if keep_source_keys {
            dictionary.keys().to_data()
        } else {
            deep_copy_array_sliced(dictionary.keys()).to_data()
        };
        let values = self.dictionary_values[dictionary_values].values.to_data();
        // SAFETY: `keys` is a value-identical logical copy of this dictionary's
        // primitive keys. `values` is a value-identical copy of the same
        // source values array, so the original key bounds and dictionary types
        // remain valid when the two are combined.
        let data = unsafe {
            keys.into_builder()
                .data_type(array.data_type().clone())
                .child_data(vec![values])
                .build_unchecked()
        };
        make_array(data)
    }

    fn record_dictionary_run(
        &mut self,
        dictionary: Option<&DictionaryArrayInfo>,
        dictionary_values: Option<usize>,
    ) {
        let (Some(dictionary), Some(dictionary_values)) = (dictionary, dictionary_values) else {
            self.dictionary_run = None;
            return;
        };
        let Some(keys) = dictionary.keys.as_ref() else {
            self.dictionary_run = None;
            return;
        };

        if self.buffered_arrays.is_empty() {
            self.dictionary_run = Some(DictionaryRun::new(
                dictionary_values,
                keys,
                dictionary.shared_bytes > self.cache_bytes,
            ));
        } else if let Some(run) = self.dictionary_run.as_mut() {
            if run.continues(
                dictionary_values,
                keys,
                dictionary.shared_bytes > self.cache_bytes,
            ) {
                run.next_key_offset = keys.end;
            } else {
                self.dictionary_run = None;
            }
        }
    }

    fn take_buffered_arrays(&mut self) -> Vec<ArrayRef> {
        self.dictionary_values.clear();
        self.dictionary_run = None;
        std::mem::take(&mut self.buffered_arrays)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{
        Array, DictionaryArray, Int8Array, Int32Array, RecordBatch, StringArray,
        types::{Int8Type, Int32Type},
    };
    use arrow_schema::{DataType, Field, Schema};
    use futures::{StreamExt, stream};

    #[test]
    fn sliced_utf8_cache_does_not_copy_the_full_parent() {
        let value = "x".repeat(1024 * 1024);
        let parent = Arc::new(StringArray::from(vec![value.as_str(); 16])) as ArrayRef;
        let slice = parent.slice(0, 2);
        assert!(array_slice_memory_size(slice.as_ref()) < 3 * 1024 * 1024);

        let mut queue = AccumulationQueue::new(8 * 1024 * 1024, 0, false);
        assert!(queue.insert(slice, 0, 2).is_none());
        let retained_bytes = queue.buffered_arrays[0].get_buffer_memory_size();

        assert!(
            retained_bytes < 3 * 1024 * 1024,
            "a 2 MiB logical slice retained {retained_bytes} bytes"
        );
    }

    #[test]
    fn sliced_dictionary_values_do_not_retain_the_full_parent() {
        let value = "x".repeat(1024 * 1024);
        let parent_values = Arc::new(StringArray::from(vec![value.as_str(); 16])) as ArrayRef;
        let values = parent_values.slice(0, 2);
        let keys = Int8Array::from(vec![0_i8, 1_i8]);
        let dictionary =
            Arc::new(DictionaryArray::<Int8Type>::try_new(keys, values).unwrap()) as ArrayRef;
        assert!(array_slice_memory_size(dictionary.as_ref()) < 3 * 1024 * 1024);
        let expected = dictionary.to_data();

        let mut queue = AccumulationQueue::new(8 * 1024 * 1024, 0, false);
        assert!(queue.insert(dictionary, 0, 2).is_none());
        assert_eq!(queue.buffered_arrays[0].to_data(), expected);
        let retained_bytes = queue.buffered_arrays[0].get_buffer_memory_size();

        assert!(
            retained_bytes < 3 * 1024 * 1024,
            "a 2 MiB dictionary retained {retained_bytes} bytes"
        );
    }

    #[test]
    fn copied_dictionary_slices_reuse_shared_values() {
        let values = Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef;
        let keys = Int8Array::from((0..100).map(|index| (index % 2) as i8).collect::<Vec<_>>());
        let dictionary =
            Arc::new(DictionaryArray::<Int8Type>::try_new(keys, values).unwrap()) as ArrayRef;

        let slice_size = array_slice_memory_size_parts(dictionary.slice(0, 10).as_ref());
        let cache_bytes = (5 * slice_size.incremental) as u64;
        let mut queue = AccumulationQueue::new(cache_bytes, 0, false);
        for offset in (0..50).step_by(10) {
            assert!(
                queue
                    .insert(dictionary.slice(offset, 10), offset as u64, 10)
                    .is_none()
            );
        }
        assert_eq!(
            queue.pending_bytes(),
            (slice_size.shared + 5 * slice_size.incremental) as u64
        );
        let (arrays, _, num_rows) = queue.flush().unwrap();
        let data = crate::data::DataBlock::from_arrays(&arrays, num_rows);
        let dictionary = data.as_dictionary().unwrap();

        assert_eq!(dictionary.dictionary.num_values(), 2);
    }

    #[test]
    fn copied_dictionary_wrappers_sharing_buffers_reuse_values() {
        let source_values = Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef;
        let arrays = (0..5)
            .map(|_| {
                let values = make_array(source_values.to_data());
                let keys =
                    Int8Array::from((0..10).map(|index| (index % 2) as i8).collect::<Vec<_>>());
                Arc::new(DictionaryArray::<Int8Type>::try_new(keys, values).unwrap()) as ArrayRef
            })
            .collect::<Vec<_>>();

        let baseline = crate::data::DataBlock::from_arrays(&arrays, 50);
        assert_eq!(baseline.as_dictionary().unwrap().dictionary.num_values(), 2);

        let mut queue = AccumulationQueue::new(8 * 1024 * 1024, 0, false);
        for (row_number, array) in arrays.into_iter().enumerate() {
            assert!(queue.insert(array, (row_number * 10) as u64, 10).is_none());
        }
        let array_size = array_slice_memory_size_parts(queue.buffered_arrays[0].as_ref());
        assert_eq!(
            queue.pending_bytes(),
            (array_size.shared + 5 * array_size.incremental) as u64
        );
        let (arrays, _, num_rows) = queue.flush().unwrap();
        let copied = crate::data::DataBlock::from_arrays(&arrays, num_rows);

        assert_eq!(copied.as_dictionary().unwrap().dictionary.num_values(), 2);
    }

    #[test]
    fn oversized_dictionary_values_do_not_disable_the_cache_bound() {
        let values = Arc::new(StringArray::from(vec!["x".repeat(2 * 1024)])) as ArrayRef;
        let mut queue = AccumulationQueue::new(1024, 0, false);
        let mut has_flushed = false;

        for row_number in 0..64 {
            let keys = Int8Array::from(vec![0_i8; 1024]);
            let dictionary =
                Arc::new(DictionaryArray::<Int8Type>::try_new(keys, values.clone()).unwrap())
                    as ArrayRef;
            if queue.insert(dictionary, row_number * 1024, 1024).is_some() {
                has_flushed = true;
                break;
            }
        }

        assert!(
            has_flushed,
            "an oversized shared dictionary retained {} bytes without ever flushing a 1024-byte cache",
            queue.pending_bytes()
        );
    }

    #[test]
    fn oversized_dictionary_values_include_structural_bytes_in_the_cache_bound() {
        let values = Arc::new(StringArray::from(vec!["x".repeat(2 * 1024)])) as ArrayRef;
        let keys = Int8Array::from(vec![0_i8]);
        let dictionary =
            Arc::new(DictionaryArray::<Int8Type>::try_new(keys, values).unwrap()) as ArrayRef;
        let mut queue = AccumulationQueue::new(1024, 0, false);

        assert!(
            queue
                .insert_with_additional_bytes(dictionary, 0, 1, 1024)
                .is_some(),
            "structural bytes must stop oversized dictionary deferral"
        );
    }

    #[tokio::test]
    async fn large_dictionary_is_not_repeated_across_checkpoint_slices() {
        const MIB: usize = 1024 * 1024;
        let value = "x".repeat(5 * MIB);
        let values = Arc::new(StringArray::from(vec![value.as_str(), value.as_str()])) as ArrayRef;
        let keys = Int32Array::from(
            (0..10 * MIB)
                .map(|index| (index % 2) as i32)
                .collect::<Vec<_>>(),
        );
        let dictionary =
            Arc::new(DictionaryArray::<Int32Type>::try_new(keys, values).unwrap()) as ArrayRef;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "item",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![dictionary]).unwrap();

        let slices = lance_arrow::stream::rechunk_stream_by_size_with_shared_estimator(
            stream::iter([Ok::<_, arrow_schema::ArrowError>(batch)]),
            schema,
            0,
            10 * MIB,
            lance_arrow::memory::batch_slice_memory_size_parts,
        )
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

        let mut queue = AccumulationQueue::new(8 * MIB as u64, 0, false);
        let mut encoded_dictionary_values = 0;
        for (row_number, slice) in slices.iter().enumerate() {
            if let Some((arrays, _, num_rows)) = queue.insert(
                slice.column(0).clone(),
                row_number as u64,
                slice.num_rows() as u64,
            ) {
                let data = crate::data::DataBlock::from_arrays(&arrays, num_rows);
                encoded_dictionary_values += data.as_dictionary().unwrap().dictionary.num_values();
            }
        }
        if let Some((arrays, _, num_rows)) = queue.flush() {
            let data = crate::data::DataBlock::from_arrays(&arrays, num_rows);
            encoded_dictionary_values += data.as_dictionary().unwrap().dictionary.num_values();
        }

        assert_eq!(slices.len(), 1);
        assert_eq!(encoded_dictionary_values, 2);
    }
}
