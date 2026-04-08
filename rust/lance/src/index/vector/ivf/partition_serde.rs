// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Serialization and zero-copy deserialization for IVF partition cache entries.
//!
//! The format is:
//!
//! ```text
//! [header_len: u64 LE]
//! [header: JSON bytes]
//! [sub_index Arrow IPC stream]
//! [... quantizer-specific IPC streams ...]
//! [storage Arrow IPC stream]
//! ```
//!
//! Each IPC section is a self-delimiting Arrow IPC stream (schema + batches + EOS
//! marker), written directly to the underlying writer without buffering. On
//! deserialization, each message is read into a per-message buffer and zero-copy
//! decoded via [`lance_arrow::ipc`].

use std::io::Write;
use std::sync::Arc;

use arrow_array::{FixedSizeListArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use lance_arrow::ipc::{
    read_ipc_stream_at, read_ipc_stream_single_at, read_len_prefixed_bytes_at, write_ipc_stream,
    write_ipc_stream_batches, write_len_prefixed_bytes,
};
use lance_core::cache::CacheCodecImpl;
use lance_core::{Error, Result};
use lance_index::vector::bq::RQRotationType;
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::bq::storage::RabitQuantizationMetadata;
use lance_index::vector::flat::index::{FlatBinQuantizer, FlatMetadata, FlatQuantizer};
use lance_index::vector::pq::ProductQuantizer;
use lance_index::vector::pq::storage::ProductQuantizationMetadata;
use lance_index::vector::quantizer::{Quantization, QuantizerStorage};
use lance_index::vector::sq::ScalarQuantizer;
use lance_index::vector::sq::storage::ScalarQuantizationMetadata;
use lance_index::vector::storage::VectorStore;
use lance_index::vector::v3::subindex::IvfSubIndex;
use lance_linalg::distance::DistanceType;
use serde::{Deserialize, Serialize};

use super::v2::PartitionEntry;

/// Returns an erased codec for `PartitionEntry<S, Q>` by matching on the
/// quantizer's [`QuantizationType`]. Returns `None` for quantizer types
/// that don't have a `CacheCodecImpl` implementation.
///
/// Uses enum dispatch rather than trait bounds to avoid propagating
/// `CacheCodecImpl` constraints through the `IVFIndex<S, Q>` type hierarchy.
pub fn partition_entry_codec<S: IvfSubIndex + 'static, Q: Quantization + 'static>()
-> Option<lance_core::cache::CacheCodec> {
    use lance_index::vector::quantizer::QuantizationType;
    match Q::quantization_type() {
        QuantizationType::Product => Some(codec_for::<S, Q, ProductQuantizer>()),
        QuantizationType::Flat => Some(codec_for::<S, Q, FlatQuantizer>()),
        QuantizationType::FlatBin => Some(codec_for::<S, Q, FlatBinQuantizer>()),
        QuantizationType::Scalar => Some(codec_for::<S, Q, ScalarQuantizer>()),
        QuantizationType::Rabit => Some(codec_for::<S, Q, RabitQuantizer>()),
    }
}

type ArcAny = Arc<dyn std::any::Any + Send + Sync>;

fn serialize_partition_entry<S, Concrete>(
    any: &ArcAny,
    writer: &mut dyn Write,
) -> lance_core::Result<()>
where
    S: IvfSubIndex + 'static,
    Concrete: Quantization + 'static,
    PartitionEntry<S, Concrete>: CacheCodecImpl,
{
    let concrete = any
        .downcast_ref::<PartitionEntry<S, Concrete>>()
        .expect("quantization_type matched but downcast failed (this is a bug)");
    concrete.serialize(writer)
}

fn deserialize_partition_entry<S, Q, Concrete>(data: &Bytes) -> lance_core::Result<ArcAny>
where
    S: IvfSubIndex + 'static,
    Q: Quantization + 'static,
    Concrete: Quantization + 'static,
    PartitionEntry<S, Concrete>: CacheCodecImpl,
{
    let concrete = PartitionEntry::<S, Concrete>::deserialize(data)?;
    let any: ArcAny = Arc::new(concrete);
    Ok(any
        .downcast::<PartitionEntry<S, Q>>()
        .expect("quantization_type matched but downcast failed (this is a bug)"))
}

/// Build a CacheCodec for `PartitionEntry<S, Q>` by delegating to the
/// CacheCodecImpl for `PartitionEntry<S, Concrete>`.
///
/// Q and Concrete must be the same type (enforced by the QuantizationType
/// match in the caller). Uses Any-based downcasting to bridge the types.
fn codec_for<
    S: IvfSubIndex + 'static,
    Q: Quantization + 'static,
    Concrete: Quantization + 'static,
>() -> lance_core::cache::CacheCodec
where
    PartitionEntry<S, Concrete>: CacheCodecImpl,
{
    lance_core::cache::CacheCodec::new(
        serialize_partition_entry::<S, Concrete>,
        deserialize_partition_entry::<S, Q, Concrete>,
    )
}

// ---------------------------------------------------------------------------
// Common helpers
// ---------------------------------------------------------------------------

fn distance_type_to_u8(dt: DistanceType) -> u8 {
    match dt {
        DistanceType::L2 => 0,
        DistanceType::Cosine => 1,
        DistanceType::Dot => 2,
        DistanceType::Hamming => 3,
    }
}

fn u8_to_distance_type(v: u8) -> Result<DistanceType> {
    match v {
        0 => Ok(DistanceType::L2),
        1 => Ok(DistanceType::Cosine),
        2 => Ok(DistanceType::Dot),
        3 => Ok(DistanceType::Hamming),
        _ => Err(Error::io(format!("unknown distance type: {v}"))),
    }
}

fn rotation_type_to_u8(rt: RQRotationType) -> u8 {
    match rt {
        RQRotationType::Matrix => 0,
        RQRotationType::Fast => 1,
    }
}

fn u8_to_rotation_type(v: u8) -> Result<RQRotationType> {
    match v {
        0 => Ok(RQRotationType::Matrix),
        1 => Ok(RQRotationType::Fast),
        _ => Err(Error::io(format!("unknown rotation type: {v}"))),
    }
}

/// Write a JSON-serializable header using [`write_len_prefixed_bytes`].
fn write_json_header(writer: &mut dyn Write, header: &impl Serialize) -> Result<()> {
    let header_json = serde_json::to_vec(header)?;
    write_len_prefixed_bytes(writer, &header_json)?;
    Ok(())
}

/// Read a JSON header written by [`write_json_header`].
fn read_json_header<T: serde::de::DeserializeOwned>(data: &Bytes, offset: &mut usize) -> Result<T> {
    let bytes = read_len_prefixed_bytes_at(data, offset).map_err(|e| Error::io(e.to_string()))?;
    serde_json::from_slice(&bytes).map_err(|e| Error::io(e.to_string()))
}

/// Wrap a `FixedSizeListArray` in a single-column `RecordBatch` with the given
/// column name.
fn fsl_to_batch(arr: &FixedSizeListArray, name: &str) -> Result<RecordBatch> {
    let field = Field::new(
        name,
        DataType::FixedSizeList(
            Arc::new(Field::new("item", arr.value_type(), true)),
            arr.value_length(),
        ),
        false,
    );
    let schema = Arc::new(Schema::new(vec![field]));
    Ok(RecordBatch::try_new(schema, vec![Arc::new(arr.clone())])?)
}

/// Extract a `FixedSizeListArray` from the first column of a `RecordBatch`.
fn batch_to_fsl(batch: &RecordBatch) -> Result<FixedSizeListArray> {
    batch
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .cloned()
        .ok_or_else(|| Error::io("column is not FixedSizeListArray".to_string()))
}

fn codebook_to_batch(codebook: &FixedSizeListArray) -> Result<RecordBatch> {
    fsl_to_batch(codebook, "codebook")
}

fn batch_to_codebook(batch: &RecordBatch) -> Result<FixedSizeListArray> {
    batch_to_fsl(batch)
}

// ---------------------------------------------------------------------------
// PQ
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct PqPartitionHeader {
    distance_type: u8,
    nbits: u32,
    num_sub_vectors: usize,
    dimension: usize,
    transposed: bool,
}

impl<S: IvfSubIndex> CacheCodecImpl for PartitionEntry<S, ProductQuantizer> {
    fn serialize(&self, writer: &mut dyn Write) -> Result<()> {
        let metadata = self.storage.metadata();
        let distance_type = self.storage.distance_type();

        let codebook = metadata.codebook.as_ref().ok_or_else(|| {
            Error::io("PQ metadata missing codebook during serialization".to_string())
        })?;

        let header = PqPartitionHeader {
            distance_type: distance_type_to_u8(distance_type),
            nbits: metadata.nbits,
            num_sub_vectors: metadata.num_sub_vectors,
            dimension: metadata.dimension,
            transposed: metadata.transposed,
        };

        write_json_header(writer, &header)?;
        write_ipc_stream(&self.index.to_batch()?, writer)?;
        write_ipc_stream(&codebook_to_batch(codebook)?, writer)?;
        write_ipc_stream_batches(self.storage.to_batches()?, writer)?;

        Ok(())
    }

    fn deserialize(data: &Bytes) -> Result<Self> {
        let mut offset = 0;
        let header: PqPartitionHeader = read_json_header(data, &mut offset)?;
        let distance_type = u8_to_distance_type(header.distance_type)?;

        let sub_index_batch =
            read_ipc_stream_single_at(data, &mut offset).map_err(|e| Error::io(e.to_string()))?;
        let codebook_batch =
            read_ipc_stream_single_at(data, &mut offset).map_err(|e| Error::io(e.to_string()))?;
        let storage_batch =
            read_ipc_stream_single_at(data, &mut offset).map_err(|e| Error::io(e.to_string()))?;

        let index = S::load(sub_index_batch)?;
        let codebook = batch_to_codebook(&codebook_batch)?;

        let metadata = ProductQuantizationMetadata {
            codebook_position: 0,
            nbits: header.nbits,
            num_sub_vectors: header.num_sub_vectors,
            dimension: header.dimension,
            codebook: Some(codebook),
            codebook_tensor: Vec::new(),
            transposed: header.transposed,
        };

        let storage = <ProductQuantizer as Quantization>::Storage::try_from_batch(
            storage_batch,
            &metadata,
            distance_type,
            None,
        )?;

        Ok(Self { index, storage })
    }
}

// ---------------------------------------------------------------------------
// Flat (Float32)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct FlatPartitionHeader {
    distance_type: u8,
    dim: usize,
}

impl<S: IvfSubIndex> CacheCodecImpl for PartitionEntry<S, FlatQuantizer> {
    fn serialize(&self, writer: &mut dyn Write) -> Result<()> {
        let metadata = self.storage.metadata();
        let distance_type = self.storage.distance_type();

        let header = FlatPartitionHeader {
            distance_type: distance_type_to_u8(distance_type),
            dim: metadata.dim,
        };

        write_json_header(writer, &header)?;
        write_ipc_stream(&self.index.to_batch()?, writer)?;
        write_ipc_stream_batches(self.storage.to_batches()?, writer)?;

        Ok(())
    }

    fn deserialize(data: &Bytes) -> Result<Self> {
        let mut offset = 0;
        let header: FlatPartitionHeader = read_json_header(data, &mut offset)?;
        let distance_type = u8_to_distance_type(header.distance_type)?;

        let sub_index_batch =
            read_ipc_stream_single_at(data, &mut offset).map_err(|e| Error::io(e.to_string()))?;
        let storage_batch =
            read_ipc_stream_single_at(data, &mut offset).map_err(|e| Error::io(e.to_string()))?;

        let index = S::load(sub_index_batch)?;
        let metadata = FlatMetadata { dim: header.dim };
        let storage = <FlatQuantizer as Quantization>::Storage::try_from_batch(
            storage_batch,
            &metadata,
            distance_type,
            None,
        )?;

        Ok(Self { index, storage })
    }
}

// ---------------------------------------------------------------------------
// Flat (Binary / Hamming)
// ---------------------------------------------------------------------------

impl<S: IvfSubIndex> CacheCodecImpl for PartitionEntry<S, FlatBinQuantizer> {
    fn serialize(&self, writer: &mut dyn Write) -> Result<()> {
        let metadata = self.storage.metadata();
        let distance_type = self.storage.distance_type();

        let header = FlatPartitionHeader {
            distance_type: distance_type_to_u8(distance_type),
            dim: metadata.dim,
        };

        write_json_header(writer, &header)?;
        write_ipc_stream(&self.index.to_batch()?, writer)?;
        write_ipc_stream_batches(self.storage.to_batches()?, writer)?;

        Ok(())
    }

    fn deserialize(data: &Bytes) -> Result<Self> {
        let mut offset = 0;
        let header: FlatPartitionHeader = read_json_header(data, &mut offset)?;
        let distance_type = u8_to_distance_type(header.distance_type)?;

        let sub_index_batch =
            read_ipc_stream_single_at(data, &mut offset).map_err(|e| Error::io(e.to_string()))?;
        let storage_batch =
            read_ipc_stream_single_at(data, &mut offset).map_err(|e| Error::io(e.to_string()))?;

        let index = S::load(sub_index_batch)?;
        let metadata = FlatMetadata { dim: header.dim };
        let storage = <FlatBinQuantizer as Quantization>::Storage::try_from_batch(
            storage_batch,
            &metadata,
            distance_type,
            None,
        )?;

        Ok(Self { index, storage })
    }
}

// ---------------------------------------------------------------------------
// SQ
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct SqPartitionHeader {
    distance_type: u8,
    num_bits: u16,
    dim: usize,
    bounds_start: f64,
    bounds_end: f64,
}

impl<S: IvfSubIndex> CacheCodecImpl for PartitionEntry<S, ScalarQuantizer> {
    fn serialize(&self, writer: &mut dyn Write) -> Result<()> {
        let metadata = self.storage.metadata();
        let distance_type = self.storage.distance_type();

        let header = SqPartitionHeader {
            distance_type: distance_type_to_u8(distance_type),
            num_bits: metadata.num_bits,
            dim: metadata.dim,
            bounds_start: metadata.bounds.start,
            bounds_end: metadata.bounds.end,
        };

        write_json_header(writer, &header)?;
        write_ipc_stream(&self.index.to_batch()?, writer)?;
        // SQ storage may contain multiple batches; stream them all in one IPC stream.
        write_ipc_stream_batches(self.storage.to_batches()?, writer)?;

        Ok(())
    }

    fn deserialize(data: &Bytes) -> Result<Self> {
        let mut offset = 0;
        let header: SqPartitionHeader = read_json_header(data, &mut offset)?;
        let distance_type = u8_to_distance_type(header.distance_type)?;

        let sub_index_batch =
            read_ipc_stream_single_at(data, &mut offset).map_err(|e| Error::io(e.to_string()))?;
        let storage_batches =
            read_ipc_stream_at(data, &mut offset).map_err(|e| Error::io(e.to_string()))?;

        let index = S::load(sub_index_batch)?;
        let metadata = ScalarQuantizationMetadata {
            dim: header.dim,
            num_bits: header.num_bits,
            bounds: header.bounds_start..header.bounds_end,
        };
        let storage = <ScalarQuantizer as Quantization>::Storage::try_new(
            metadata.num_bits,
            distance_type,
            metadata.bounds,
            storage_batches,
            None,
        )?;

        Ok(Self { index, storage })
    }
}

// ---------------------------------------------------------------------------
// RabitQ
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct RabitPartitionHeader {
    distance_type: u8,
    num_bits: u8,
    code_dim: u32,
    /// 0 = Matrix, 1 = Fast
    rotation_type: u8,
    /// Fast rotation signs (only set when rotation_type == Fast).
    fast_rotation_signs: Option<Vec<u8>>,
}

impl<S: IvfSubIndex> CacheCodecImpl for PartitionEntry<S, RabitQuantizer> {
    fn serialize(&self, writer: &mut dyn Write) -> Result<()> {
        let metadata = self.storage.metadata();
        let distance_type = self.storage.distance_type();

        let header = RabitPartitionHeader {
            distance_type: distance_type_to_u8(distance_type),
            num_bits: metadata.num_bits,
            code_dim: metadata.code_dim,
            rotation_type: rotation_type_to_u8(metadata.rotation_type),
            fast_rotation_signs: metadata.fast_rotation_signs.clone(),
        };

        write_json_header(writer, &header)?;

        write_ipc_stream(&self.index.to_batch()?, writer)?;

        // Write the rotation matrix IPC stream only for Matrix rotation; the
        // Fast rotation case stores its signs compactly in the JSON header.
        if metadata.rotation_type == RQRotationType::Matrix {
            let mat = metadata.rotate_mat.as_ref().ok_or_else(|| {
                Error::io(
                    "RabitQ Matrix metadata missing rotate_mat during serialization".to_string(),
                )
            })?;
            write_ipc_stream(&fsl_to_batch(mat, "rotate_mat")?, writer)?;
        }

        write_ipc_stream_batches(self.storage.to_batches()?, writer)?;

        Ok(())
    }

    fn deserialize(data: &Bytes) -> Result<Self> {
        let mut offset = 0;
        let header: RabitPartitionHeader = read_json_header(data, &mut offset)?;
        let distance_type = u8_to_distance_type(header.distance_type)?;
        let rotation_type = u8_to_rotation_type(header.rotation_type)?;

        let sub_index_batch =
            read_ipc_stream_single_at(data, &mut offset).map_err(|e| Error::io(e.to_string()))?;

        let rotate_mat = if rotation_type == RQRotationType::Matrix {
            let mat_batch = read_ipc_stream_single_at(data, &mut offset)
                .map_err(|e| Error::io(e.to_string()))?;
            Some(batch_to_fsl(&mat_batch)?)
        } else {
            None
        };

        let storage_batch =
            read_ipc_stream_single_at(data, &mut offset).map_err(|e| Error::io(e.to_string()))?;

        let index = S::load(sub_index_batch)?;
        let metadata = RabitQuantizationMetadata {
            rotate_mat,
            rotate_mat_position: None,
            fast_rotation_signs: header.fast_rotation_signs,
            rotation_type,
            code_dim: header.code_dim,
            num_bits: header.num_bits,
            // The storage batch already has packed codes; skip re-packing.
            packed: true,
        };
        let storage = <RabitQuantizer as Quantization>::Storage::try_from_batch(
            storage_batch,
            &metadata,
            distance_type,
            None,
        )?;

        Ok(Self { index, storage })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::{
        Float32Array, UInt8Array, UInt64Array,
        types::{Float32Type, UInt8Type},
    };
    use arrow_schema::{DataType, Field, Schema};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::vector::bq::storage::RABIT_CODE_COLUMN;
    use lance_index::vector::bq::transform::{ADD_FACTORS_COLUMN, SCALE_FACTORS_COLUMN};
    use lance_index::vector::bq::{RQRotationType, builder::RabitQuantizer};
    use lance_index::vector::flat::index::FlatIndex;
    use lance_index::vector::flat::storage::FlatFloatStorage;
    use lance_index::vector::sq::storage::ScalarQuantizationStorage;

    // ----- PQ helpers -------------------------------------------------------

    fn make_test_codebook(dim: usize, num_sub_vectors: usize) -> FixedSizeListArray {
        let sub_dim = dim / num_sub_vectors;
        let num_centroids = 256;
        let total_values = num_sub_vectors * num_centroids * sub_dim;
        let values: Vec<f32> = (0..total_values).map(|i| i as f32 * 0.01).collect();
        let values_array = Float32Array::from(values);
        FixedSizeListArray::try_new_from_values(values_array, sub_dim as i32).unwrap()
    }

    fn make_test_pq_storage(
        num_rows: usize,
        dim: usize,
        num_sub_vectors: usize,
    ) -> <ProductQuantizer as Quantization>::Storage {
        let codebook = make_test_codebook(dim, num_sub_vectors);
        let row_ids = UInt64Array::from((0..num_rows as u64).collect::<Vec<_>>());
        let pq_codes_flat: Vec<u8> = (0..num_rows * num_sub_vectors)
            .map(|i| (i % 256) as u8)
            .collect();
        let pq_codes = UInt8Array::from(pq_codes_flat);
        let pq_codes_fsl =
            FixedSizeListArray::try_new_from_values(pq_codes, num_sub_vectors as i32).unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new(lance_core::ROW_ID, DataType::UInt64, false),
            Field::new(
                lance_index::vector::PQ_CODE_COLUMN,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::UInt8, true)),
                    num_sub_vectors as i32,
                ),
                false,
            ),
        ]));

        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(row_ids), Arc::new(pq_codes_fsl)]).unwrap();

        <ProductQuantizer as Quantization>::Storage::new(
            codebook,
            batch,
            8,
            num_sub_vectors,
            dim,
            DistanceType::L2,
            false,
            None,
        )
        .unwrap()
    }

    // ----- PQ tests ---------------------------------------------------------

    #[test]
    fn test_roundtrip_flat_pq() {
        let dim = 128;
        let num_sub_vectors = 16;
        let num_rows = 100;

        let storage = make_test_pq_storage(num_rows, dim, num_sub_vectors);
        let entry = PartitionEntry::<FlatIndex, ProductQuantizer> {
            index: FlatIndex::default(),
            storage,
        };

        let mut serialized = Vec::new();
        entry.serialize(&mut serialized).unwrap();
        let deserialized = PartitionEntry::<FlatIndex, ProductQuantizer>::deserialize(
            &bytes::Bytes::from(serialized),
        )
        .unwrap();

        assert_eq!(entry.storage, deserialized.storage);
    }

    #[test]
    fn test_roundtrip_preserves_distance_type() {
        for dt in [DistanceType::L2, DistanceType::Cosine, DistanceType::Dot] {
            let dim = 32;
            let num_sub_vectors = 4;
            let codebook = make_test_codebook(dim, num_sub_vectors);
            let row_ids = UInt64Array::from(vec![0u64, 1, 2]);
            let pq_codes = UInt8Array::from(vec![0u8; 3 * num_sub_vectors]);
            let pq_codes_fsl =
                FixedSizeListArray::try_new_from_values(pq_codes, num_sub_vectors as i32).unwrap();

            let schema = Arc::new(Schema::new(vec![
                Field::new(lance_core::ROW_ID, DataType::UInt64, false),
                Field::new(
                    lance_index::vector::PQ_CODE_COLUMN,
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::UInt8, true)),
                        num_sub_vectors as i32,
                    ),
                    false,
                ),
            ]));
            let batch =
                RecordBatch::try_new(schema, vec![Arc::new(row_ids), Arc::new(pq_codes_fsl)])
                    .unwrap();

            let storage = <ProductQuantizer as Quantization>::Storage::new(
                codebook,
                batch,
                8,
                num_sub_vectors,
                dim,
                dt,
                false,
                None,
            )
            .unwrap();

            let entry = PartitionEntry::<FlatIndex, ProductQuantizer> {
                index: FlatIndex::default(),
                storage,
            };

            let mut bytes = Vec::new();
            entry.serialize(&mut bytes).unwrap();
            let restored = PartitionEntry::<FlatIndex, ProductQuantizer>::deserialize(
                &bytes::Bytes::from(bytes),
            )
            .unwrap();
            assert_eq!(
                restored.storage.distance_type(),
                entry.storage.distance_type()
            );
        }
    }

    #[test]
    fn test_empty_partition() {
        let dim = 16;
        let num_sub_vectors = 2;
        let storage = make_test_pq_storage(0, dim, num_sub_vectors);
        let entry = PartitionEntry::<FlatIndex, ProductQuantizer> {
            index: FlatIndex::default(),
            storage,
        };

        let mut serialized = Vec::new();
        entry.serialize(&mut serialized).unwrap();
        let deserialized = PartitionEntry::<FlatIndex, ProductQuantizer>::deserialize(
            &bytes::Bytes::from(serialized),
        )
        .unwrap();
        assert_eq!(entry.storage, deserialized.storage);
    }

    #[test]
    fn test_truncated_data_errors() {
        // Serialize a valid entry, then truncate the bytes and verify that
        // deserialization fails rather than panicking.
        let storage = make_test_pq_storage(1, 16, 2);
        let entry = PartitionEntry::<FlatIndex, ProductQuantizer> {
            index: FlatIndex::default(),
            storage,
        };
        let mut bytes = Vec::new();
        entry.serialize(&mut bytes).unwrap();
        bytes.truncate(3);
        assert!(
            PartitionEntry::<FlatIndex, ProductQuantizer>::deserialize(&bytes::Bytes::from(bytes))
                .is_err()
        );
    }

    // ----- Flat helpers -----------------------------------------------------

    fn make_flat_storage(num_rows: usize, dim: usize) -> FlatFloatStorage {
        let values: Vec<f32> = (0..num_rows * dim).map(|i| i as f32 * 0.01).collect();
        let values_array = Float32Array::from(values);
        let vectors = FixedSizeListArray::try_new_from_values(values_array, dim as i32).unwrap();
        FlatFloatStorage::new(vectors, DistanceType::L2)
    }

    // ----- Flat tests -------------------------------------------------------

    #[test]
    fn test_roundtrip_flat_flat() {
        let storage = make_flat_storage(50, 64);
        let entry = PartitionEntry::<FlatIndex, FlatQuantizer> {
            index: FlatIndex::default(),
            storage,
        };

        let mut bytes = Vec::new();
        entry.serialize(&mut bytes).unwrap();
        let restored =
            PartitionEntry::<FlatIndex, FlatQuantizer>::deserialize(&bytes::Bytes::from(bytes))
                .unwrap();

        assert_eq!(
            restored.storage.metadata().dim,
            entry.storage.metadata().dim
        );
        assert_eq!(
            restored.storage.distance_type(),
            entry.storage.distance_type()
        );
        assert_eq!(restored.storage.len(), entry.storage.len());
        let orig_batch = entry.storage.to_batches().unwrap().next().unwrap();
        let rest_batch = restored.storage.to_batches().unwrap().next().unwrap();
        assert_eq!(orig_batch, rest_batch);
    }

    #[test]
    fn test_flat_distance_types() {
        for dt in [DistanceType::L2, DistanceType::Cosine, DistanceType::Dot] {
            let values = Float32Array::from(vec![1.0f32; 32]);
            let vectors = FixedSizeListArray::try_new_from_values(values, 32).unwrap();
            let storage = FlatFloatStorage::new(vectors, dt);
            let entry = PartitionEntry::<FlatIndex, FlatQuantizer> {
                index: FlatIndex::default(),
                storage,
            };
            let mut bytes = Vec::new();
            entry.serialize(&mut bytes).unwrap();
            let restored =
                PartitionEntry::<FlatIndex, FlatQuantizer>::deserialize(&bytes::Bytes::from(bytes))
                    .unwrap();
            assert_eq!(restored.storage.distance_type(), dt);
        }
    }

    // ----- SQ helpers -------------------------------------------------------

    fn make_sq_storage(
        num_rows: usize,
        dim: usize,
        distance_type: DistanceType,
    ) -> ScalarQuantizationStorage {
        let row_ids = UInt64Array::from_iter_values(0..num_rows as u64);
        let sq_codes_flat: Vec<u8> = (0..num_rows * dim).map(|i| (i % 256) as u8).collect();
        let sq_codes = UInt8Array::from(sq_codes_flat);
        let sq_codes_fsl = FixedSizeListArray::try_new_from_values(sq_codes, dim as i32).unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new(lance_core::ROW_ID, DataType::UInt64, false),
            Field::new(
                lance_index::vector::SQ_CODE_COLUMN,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::UInt8, true)),
                    dim as i32,
                ),
                false,
            ),
        ]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(row_ids), Arc::new(sq_codes_fsl)]).unwrap();

        ScalarQuantizationStorage::try_new(8, distance_type, -1.0..1.0, [batch], None).unwrap()
    }

    // ----- SQ tests ---------------------------------------------------------

    #[test]
    fn test_roundtrip_flat_sq() {
        let storage = make_sq_storage(100, 64, DistanceType::L2);
        let entry = PartitionEntry::<FlatIndex, ScalarQuantizer> {
            index: FlatIndex::default(),
            storage,
        };

        let mut bytes = Vec::new();
        entry.serialize(&mut bytes).unwrap();
        let restored =
            PartitionEntry::<FlatIndex, ScalarQuantizer>::deserialize(&bytes::Bytes::from(bytes))
                .unwrap();

        let m = entry.storage.metadata();
        let rm = restored.storage.metadata();
        assert_eq!(rm.dim, m.dim);
        assert_eq!(rm.num_bits, m.num_bits);
        assert_eq!(rm.bounds, m.bounds);
        assert_eq!(
            restored.storage.distance_type(),
            entry.storage.distance_type()
        );
        assert_eq!(restored.storage.len(), entry.storage.len());

        let orig_ids: Vec<u64> = entry.storage.row_ids().copied().collect();
        let rest_ids: Vec<u64> = restored.storage.row_ids().copied().collect();
        assert_eq!(orig_ids, rest_ids);
    }

    #[test]
    fn test_sq_distance_types() {
        for dt in [DistanceType::L2, DistanceType::Cosine, DistanceType::Dot] {
            let storage = make_sq_storage(10, 16, dt);
            let entry = PartitionEntry::<FlatIndex, ScalarQuantizer> {
                index: FlatIndex::default(),
                storage,
            };
            let mut bytes = Vec::new();
            entry.serialize(&mut bytes).unwrap();
            let restored = PartitionEntry::<FlatIndex, ScalarQuantizer>::deserialize(
                &bytes::Bytes::from(bytes),
            )
            .unwrap();
            assert_eq!(restored.storage.distance_type(), dt);
        }
    }

    #[test]
    fn test_sq_multiple_chunks_no_copy() {
        // Build SQ storage with multiple chunks by appending batches separately.
        let dim = 16usize;
        let make_batch = |start: u64, n: usize| {
            let row_ids = UInt64Array::from_iter_values(start..start + n as u64);
            let codes = UInt8Array::from(vec![0u8; n * dim]);
            let fsl = FixedSizeListArray::try_new_from_values(codes, dim as i32).unwrap();
            let schema = Arc::new(Schema::new(vec![
                Field::new(lance_core::ROW_ID, DataType::UInt64, false),
                Field::new(
                    lance_index::vector::SQ_CODE_COLUMN,
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::UInt8, true)),
                        dim as i32,
                    ),
                    false,
                ),
            ]));
            RecordBatch::try_new(schema, vec![Arc::new(row_ids), Arc::new(fsl)]).unwrap()
        };
        // Three chunks with 10 rows each.
        let storage = ScalarQuantizationStorage::try_new(
            8,
            DistanceType::L2,
            -1.0..1.0,
            [make_batch(0, 10), make_batch(10, 10), make_batch(20, 10)],
            None,
        )
        .unwrap();
        assert_eq!(storage.len(), 30);

        let entry = PartitionEntry::<FlatIndex, ScalarQuantizer> {
            index: FlatIndex::default(),
            storage,
        };
        let mut bytes = Vec::new();
        entry.serialize(&mut bytes).unwrap();
        let restored =
            PartitionEntry::<FlatIndex, ScalarQuantizer>::deserialize(&bytes::Bytes::from(bytes))
                .unwrap();

        assert_eq!(restored.storage.len(), 30);
        let orig_ids: Vec<u64> = entry.storage.row_ids().copied().collect();
        let rest_ids: Vec<u64> = restored.storage.row_ids().copied().collect();
        assert_eq!(orig_ids, rest_ids);
    }

    // ----- RabitQ helpers ---------------------------------------------------

    fn make_rabit_storage_fast(
        num_rows: usize,
        code_dim: usize,
        distance_type: DistanceType,
    ) -> <RabitQuantizer as Quantization>::Storage {
        use lance_arrow::FixedSizeListArrayExt;

        let quantizer = RabitQuantizer::new_with_rotation::<Float32Type>(
            1,
            code_dim as i32,
            RQRotationType::Fast,
        );
        let values: Vec<f32> = (0..num_rows * code_dim)
            .map(|i| (i % 100) as f32 / 100.0 - 0.5)
            .collect();
        let values_arr = Float32Array::from(values);
        let vectors = FixedSizeListArray::try_new_from_values(values_arr, code_dim as i32).unwrap();
        let codes = quantizer
            .quantize(&vectors)
            .unwrap()
            .as_fixed_size_list()
            .clone();

        let metadata = quantizer.metadata(None);
        let batch = RecordBatch::try_from_iter(vec![
            (
                lance_core::ROW_ID,
                Arc::new(UInt64Array::from_iter_values(0..num_rows as u64))
                    as Arc<dyn arrow_array::Array>,
            ),
            (
                RABIT_CODE_COLUMN,
                Arc::new(codes) as Arc<dyn arrow_array::Array>,
            ),
            (
                ADD_FACTORS_COLUMN,
                Arc::new(Float32Array::from_iter_values(
                    (0..num_rows).map(|i| i as f32 * 0.1),
                )) as Arc<dyn arrow_array::Array>,
            ),
            (
                SCALE_FACTORS_COLUMN,
                Arc::new(Float32Array::from_iter_values(
                    (0..num_rows).map(|i| i as f32 * 0.01 + 0.5),
                )) as Arc<dyn arrow_array::Array>,
            ),
        ])
        .unwrap();

        <RabitQuantizer as Quantization>::Storage::try_from_batch(
            batch,
            &metadata,
            distance_type,
            None,
        )
        .unwrap()
    }

    // ----- RabitQ tests -----------------------------------------------------

    #[test]
    fn test_roundtrip_flat_rabitq_fast() {
        let num_rows = 50;
        let code_dim = 64;
        let storage = make_rabit_storage_fast(num_rows, code_dim, DistanceType::L2);
        let entry = PartitionEntry::<FlatIndex, RabitQuantizer> {
            index: FlatIndex::default(),
            storage,
        };

        let mut bytes = Vec::new();
        entry.serialize(&mut bytes).unwrap();
        let restored =
            PartitionEntry::<FlatIndex, RabitQuantizer>::deserialize(&bytes::Bytes::from(bytes))
                .unwrap();

        let m = entry.storage.metadata();
        let rm = restored.storage.metadata();
        assert_eq!(rm.num_bits, m.num_bits);
        assert_eq!(rm.code_dim, m.code_dim);
        assert_eq!(rm.rotation_type, m.rotation_type);
        assert_eq!(rm.fast_rotation_signs, m.fast_rotation_signs);
        assert!(rm.packed);
        assert_eq!(
            restored.storage.distance_type(),
            entry.storage.distance_type()
        );
        assert_eq!(restored.storage.len(), entry.storage.len());

        let orig_ids: Vec<u64> = entry.storage.row_ids().copied().collect();
        let rest_ids: Vec<u64> = restored.storage.row_ids().copied().collect();
        assert_eq!(orig_ids, rest_ids);

        let orig_batch = entry.storage.to_batches().unwrap().next().unwrap();
        let rest_batch = restored.storage.to_batches().unwrap().next().unwrap();
        let orig_codes = orig_batch[RABIT_CODE_COLUMN].as_fixed_size_list();
        let rest_codes = rest_batch[RABIT_CODE_COLUMN].as_fixed_size_list();
        assert_eq!(
            orig_codes.values().as_primitive::<UInt8Type>().values(),
            rest_codes.values().as_primitive::<UInt8Type>().values(),
        );
    }

    #[test]
    fn test_rabitq_distance_types() {
        for dt in [DistanceType::L2, DistanceType::Cosine, DistanceType::Dot] {
            let storage = make_rabit_storage_fast(10, 32, dt);
            let entry = PartitionEntry::<FlatIndex, RabitQuantizer> {
                index: FlatIndex::default(),
                storage,
            };
            let mut bytes = Vec::new();
            entry.serialize(&mut bytes).unwrap();
            let restored = PartitionEntry::<FlatIndex, RabitQuantizer>::deserialize(
                &bytes::Bytes::from(bytes),
            )
            .unwrap();
            assert_eq!(restored.storage.distance_type(), dt);
        }
    }

    #[test]
    fn test_ivf_index_state_roundtrip() {
        use crate::index::vector::ivf::v2::{IvfIndexState, IvfStateEntryBox};
        use lance_index::vector::flat::index::FlatQuantizer;
        use lance_index::vector::ivf::storage::IvfModel;
        use lance_index::vector::quantizer::QuantizationType;
        use lance_index::vector::v3::subindex::SubIndexType;

        // Build a minimal IvfModel (single centroid, dim=2).
        let centroids =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0f32, 1.0]), 2)
                .unwrap();
        let ivf = IvfModel::new(centroids, None);

        let state = IvfIndexState::<FlatQuantizer> {
            index_file_path: "my/index.lance".to_string(),
            uuid: "test-uuid-1234".to_string(),
            ivf: ivf.clone(),
            aux_ivf: ivf,
            distance_type: DistanceType::L2,
            sub_index_metadata: vec!["meta1".to_string()],
            metadata: lance_index::vector::flat::index::FlatMetadata { dim: 2 },
            sub_index_type: SubIndexType::Flat,
            quantization_type: QuantizationType::Flat,
            cache_key_prefix: "prefix/".to_string(),
            index_file_size: 1024,
            aux_file_size: 512,
        };

        let entry = IvfStateEntryBox(Arc::new(state));

        let mut bytes = Vec::new();
        CacheCodecImpl::serialize(&entry, &mut bytes).unwrap();

        let restored =
            <IvfStateEntryBox as CacheCodecImpl>::deserialize(&bytes::Bytes::from(bytes.clone()))
                .unwrap();

        // Re-serialize the restored entry and compare bytes — a stronger check
        // than field-by-field comparison and avoids needing to downcast.
        let mut restored_bytes = Vec::new();
        CacheCodecImpl::serialize(&restored, &mut restored_bytes).unwrap();
        assert_eq!(bytes, restored_bytes);
    }
}
