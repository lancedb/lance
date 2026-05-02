// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::BTreeMap;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    ArrowPrimitiveType, Date32Type, Date64Type, Float32Type, Float64Type, Int8Type, Int16Type,
    Int32Type, Int64Type, Time32MillisecondType, Time32SecondType, Time64MicrosecondType,
    Time64NanosecondType, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{
    Array, BinaryArray, FixedSizeBinaryArray, LargeBinaryArray, LargeStringArray, RecordBatch,
    StringArray,
};
use arrow_schema::DataType;
use lance_core::{Error, Result};

use crate::take_rows;

pub struct Partitioner {
    partition_count: u32,
    primary_key_columns: Vec<String>,
}

pub fn assigned_consumer_for_partition<'a>(
    group_id: &str,
    partition_id: u32,
    consumer_ids: &'a [String],
) -> Option<&'a str> {
    let mut best = consumer_ids.first()?;
    let mut best_score = consumer_assignment_score(group_id, partition_id, best);
    for consumer_id in &consumer_ids[1..] {
        let score = consumer_assignment_score(group_id, partition_id, consumer_id);
        if score > best_score || (score == best_score && consumer_id < best) {
            best = consumer_id;
            best_score = score;
        }
    }
    Some(best.as_str())
}

impl Partitioner {
    pub fn new(partition_count: u32, primary_key_columns: Vec<String>) -> Result<Self> {
        if partition_count == 0 {
            return Err(Error::invalid_input(
                "partition_count must be greater than 0",
            ));
        }
        if primary_key_columns.is_empty() {
            return Err(Error::invalid_input(
                "topic producer requires unenforced primary key columns",
            ));
        }
        Ok(Self {
            partition_count,
            primary_key_columns,
        })
    }

    pub fn partition_batch(&self, batch: &RecordBatch) -> Result<Vec<(u32, RecordBatch)>> {
        if self.partition_count == 1 {
            return Ok(vec![(0, batch.clone())]);
        }

        let mut row_indices_by_partition = BTreeMap::<u32, Vec<u32>>::new();
        let key_columns = self.key_columns(batch)?;
        for row_idx in 0..batch.num_rows() {
            let partition_id = self.partition_for_row(&key_columns, row_idx)?;
            row_indices_by_partition
                .entry(partition_id)
                .or_default()
                .push(row_idx as u32);
        }

        row_indices_by_partition
            .into_iter()
            .map(|(partition_id, row_indices)| {
                let partition_batch = take_rows(batch, &row_indices)?;
                Ok((partition_id, partition_batch))
            })
            .collect()
    }

    fn key_columns<'a>(&self, batch: &'a RecordBatch) -> Result<Vec<&'a dyn Array>> {
        self.primary_key_columns
            .iter()
            .map(|column| {
                batch
                    .column_by_name(column)
                    .map(|array| array.as_ref())
                    .ok_or_else(|| {
                        Error::invalid_input(format!(
                            "primary key column '{}' does not exist in record batch",
                            column
                        ))
                    })
            })
            .collect()
    }

    fn partition_for_row(&self, key_columns: &[&dyn Array], row_idx: usize) -> Result<u32> {
        let mut hasher = StableHasher::new();
        for column in key_columns {
            if row_idx >= column.len() {
                return Err(Error::invalid_input(format!(
                    "row index {} is out of range for primary key column with {} rows",
                    row_idx,
                    column.len()
                )));
            }
            hash_value(*column, row_idx, &mut hasher)?;
        }
        Ok(non_negative_murmur_bucket(hasher.finish()) % self.partition_count)
    }
}

fn consumer_assignment_score(group_id: &str, partition_id: u32, consumer_id: &str) -> u32 {
    let mut hasher = StableHasher::new();
    hasher.write_bytes(b"lance_topic_consumer_assignment_v1");
    hash_len_prefixed(group_id.as_bytes(), &mut hasher);
    hasher.write_bytes(&partition_id.to_le_bytes());
    hash_len_prefixed(consumer_id.as_bytes(), &mut hasher);
    hasher.finish()
}

struct StableHasher {
    bytes: Vec<u8>,
}

impl StableHasher {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    fn write_u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn write_bytes(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    fn finish(&self) -> u32 {
        murmur3_x86_32(&self.bytes, 0)
    }
}

fn non_negative_murmur_bucket(hash: u32) -> u32 {
    let signed = hash as i32;
    signed.unsigned_abs()
}

pub fn murmur3_x86_32(bytes: &[u8], seed: u32) -> u32 {
    const C1: u32 = 0xcc9e2d51;
    const C2: u32 = 0x1b873593;

    let mut hash = seed;
    let chunks = bytes.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let mut k = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        k = k.wrapping_mul(C1);
        k = k.rotate_left(15);
        k = k.wrapping_mul(C2);

        hash ^= k;
        hash = hash.rotate_left(13);
        hash = hash.wrapping_mul(5).wrapping_add(0xe6546b64);
    }

    let mut k = 0_u32;
    match remainder.len() {
        3 => {
            k ^= (remainder[2] as u32) << 16;
            k ^= (remainder[1] as u32) << 8;
            k ^= remainder[0] as u32;
        }
        2 => {
            k ^= (remainder[1] as u32) << 8;
            k ^= remainder[0] as u32;
        }
        1 => {
            k ^= remainder[0] as u32;
        }
        _ => {}
    }
    if !remainder.is_empty() {
        k = k.wrapping_mul(C1);
        k = k.rotate_left(15);
        k = k.wrapping_mul(C2);
        hash ^= k;
    }

    hash ^= bytes.len() as u32;
    hash ^= hash >> 16;
    hash = hash.wrapping_mul(0x85ebca6b);
    hash ^= hash >> 13;
    hash = hash.wrapping_mul(0xc2b2ae35);
    hash ^= hash >> 16;
    hash
}

fn hash_value(array: &dyn Array, row_idx: usize, hasher: &mut StableHasher) -> Result<()> {
    if array.is_null(row_idx) {
        hasher.write_u8(0);
        return Ok(());
    }

    hasher.write_u8(1);
    match array.data_type() {
        DataType::Boolean => {
            hasher.write_u8(u8::from(array.as_boolean().value(row_idx)));
        }
        DataType::Int8 => hash_primitive::<Int8Type>(array, row_idx, hasher)?,
        DataType::Int16 => hash_primitive::<Int16Type>(array, row_idx, hasher)?,
        DataType::Int32 => hash_primitive::<Int32Type>(array, row_idx, hasher)?,
        DataType::Int64 => hash_primitive::<Int64Type>(array, row_idx, hasher)?,
        DataType::UInt8 => hash_primitive::<UInt8Type>(array, row_idx, hasher)?,
        DataType::UInt16 => hash_primitive::<UInt16Type>(array, row_idx, hasher)?,
        DataType::UInt32 => hash_primitive::<UInt32Type>(array, row_idx, hasher)?,
        DataType::UInt64 => hash_primitive::<UInt64Type>(array, row_idx, hasher)?,
        DataType::Float32 => hash_primitive::<Float32Type>(array, row_idx, hasher)?,
        DataType::Float64 => hash_primitive::<Float64Type>(array, row_idx, hasher)?,
        DataType::Date32 => hash_primitive::<Date32Type>(array, row_idx, hasher)?,
        DataType::Date64 => hash_primitive::<Date64Type>(array, row_idx, hasher)?,
        DataType::Time32(arrow_schema::TimeUnit::Second) => {
            hash_primitive::<Time32SecondType>(array, row_idx, hasher)?
        }
        DataType::Time32(arrow_schema::TimeUnit::Millisecond) => {
            hash_primitive::<Time32MillisecondType>(array, row_idx, hasher)?
        }
        DataType::Time64(arrow_schema::TimeUnit::Microsecond) => {
            hash_primitive::<Time64MicrosecondType>(array, row_idx, hasher)?
        }
        DataType::Time64(arrow_schema::TimeUnit::Nanosecond) => {
            hash_primitive::<Time64NanosecondType>(array, row_idx, hasher)?
        }
        DataType::Timestamp(arrow_schema::TimeUnit::Second, _) => {
            hash_primitive::<TimestampSecondType>(array, row_idx, hasher)?
        }
        DataType::Timestamp(arrow_schema::TimeUnit::Millisecond, _) => {
            hash_primitive::<TimestampMillisecondType>(array, row_idx, hasher)?
        }
        DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, _) => {
            hash_primitive::<TimestampMicrosecondType>(array, row_idx, hasher)?
        }
        DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, _) => {
            hash_primitive::<TimestampNanosecondType>(array, row_idx, hasher)?
        }
        DataType::Utf8 => {
            let values = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    Error::internal("failed to downcast Utf8 primary key column".to_string())
                })?;
            hash_len_prefixed(values.value(row_idx).as_bytes(), hasher);
        }
        DataType::LargeUtf8 => {
            let values = array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| {
                    Error::internal("failed to downcast LargeUtf8 primary key column".to_string())
                })?;
            hash_len_prefixed(values.value(row_idx).as_bytes(), hasher);
        }
        DataType::Binary => {
            let values = array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or_else(|| {
                    Error::internal("failed to downcast Binary primary key column".to_string())
                })?;
            hash_len_prefixed(values.value(row_idx), hasher);
        }
        DataType::LargeBinary => {
            let values = array
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .ok_or_else(|| {
                    Error::internal("failed to downcast LargeBinary primary key column".to_string())
                })?;
            hash_len_prefixed(values.value(row_idx), hasher);
        }
        DataType::FixedSizeBinary(_) => {
            let values = array
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .ok_or_else(|| {
                    Error::internal(
                        "failed to downcast FixedSizeBinary primary key column".to_string(),
                    )
                })?;
            hash_len_prefixed(values.value(row_idx), hasher);
        }
        other => {
            return Err(Error::invalid_input(format!(
                "unsupported primary key data type for topic partitioning: {}",
                other
            )));
        }
    }

    Ok(())
}

fn hash_primitive<T>(array: &dyn Array, row_idx: usize, hasher: &mut StableHasher) -> Result<()>
where
    T: ArrowPrimitiveType,
    T::Native: HashNative,
{
    let values = array.as_primitive::<T>();
    values.value(row_idx).hash_to(hasher);
    Ok(())
}

fn hash_len_prefixed(bytes: &[u8], hasher: &mut StableHasher) {
    hasher.write_bytes(&(bytes.len() as u64).to_le_bytes());
    hasher.write_bytes(bytes);
}

trait HashNative {
    fn hash_to(self, hasher: &mut StableHasher);
}

macro_rules! impl_hash_native_int {
    ($($ty:ty),*) => {
        $(
            impl HashNative for $ty {
                fn hash_to(self, hasher: &mut StableHasher) {
                    hasher.write_bytes(&self.to_le_bytes());
                }
            }
        )*
    };
}

impl_hash_native_int!(i8, i16, i32, i64, u8, u16, u32, u64);

impl HashNative for f32 {
    fn hash_to(self, hasher: &mut StableHasher) {
        let canonical = if self == 0.0 {
            0.0
        } else if self.is_nan() {
            Self::NAN
        } else {
            self
        };
        hasher.write_bytes(&canonical.to_bits().to_le_bytes());
    }
}

impl HashNative for f64 {
    fn hash_to(self, hasher: &mut StableHasher) {
        let canonical = if self == 0.0 {
            0.0
        } else if self.is_nan() {
            Self::NAN
        } else {
            self
        };
        hasher.write_bytes(&canonical.to_bits().to_le_bytes());
    }
}
