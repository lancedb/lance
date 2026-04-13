// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Protobuf serialization for [`ANNIvfPartitionExec`] and [`ANNIvfSubIndexExec`].
//!
//! Proto message definitions live in `lance-datafusion` (see `pb`).
//! Conversion functions live here because they need access to `ANNIvfPartitionExec`,
//! `ANNIvfSubIndexExec`, and `Dataset`, which are defined in this crate.
//!
//! A DataFusion `PhysicalExtensionCodec` can call these functions in `try_encode`
//! and `try_decode` to support distributed execution (planner → executor).

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{Field, Schema as ArrowSchema};
use lance_core::{Error, Result};
use lance_datafusion::pb;
use lance_index::vector::Query;
use lance_linalg::distance::DistanceType;
use lance_table::format::IndexMetadata;
use lance_table::format::pb as table_pb;
use prost::Message;

use crate::Dataset;

use super::knn::{ANNIvfPartitionExec, ANNIvfSubIndexExec};
use super::table_identifier::{open_dataset_from_table_identifier, table_identifier_from_dataset};
use super::utils::PreFilterSource;

// =============================================================================
// VectorQueryProto helpers
// =============================================================================

/// Serialize an Arrow array to IPC file-format bytes.
///
/// Wraps the array in a single-column RecordBatch so that the IPC format
/// preserves the full data type (Float16, Float32, Float64, UInt8, etc.).
fn array_to_ipc_bytes(array: &dyn arrow_array::Array) -> Result<Vec<u8>> {
    let field = Field::new("key", array.data_type().clone(), true);
    let schema = Arc::new(ArrowSchema::new(vec![field]));
    let batch = RecordBatch::try_new(schema, vec![arrow_array::make_array(array.to_data())])
        .map_err(|e| Error::internal(format!("Failed to create RecordBatch for query key: {e}")))?;

    let mut buf = Vec::new();
    {
        let mut writer = arrow_ipc::writer::FileWriter::try_new(&mut buf, &batch.schema())
            .map_err(|e| Error::internal(format!("Failed to create IPC writer: {e}")))?;
        writer
            .write(&batch)
            .map_err(|e| Error::internal(format!("Failed to write IPC batch: {e}")))?;
        writer
            .finish()
            .map_err(|e| Error::internal(format!("Failed to finish IPC writer: {e}")))?;
    }
    Ok(buf)
}

/// Deserialize an Arrow array from IPC file-format bytes.
fn array_from_ipc_bytes(bytes: &[u8]) -> Result<arrow_array::ArrayRef> {
    let cursor = std::io::Cursor::new(bytes);
    let reader = arrow_ipc::reader::FileReader::try_new(cursor, None)
        .map_err(|e| Error::internal(format!("Failed to create IPC reader: {e}")))?;

    let batches: Vec<RecordBatch> = reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| Error::internal(format!("Failed to read IPC batches: {e}")))?;

    if batches.is_empty() || batches[0].num_columns() == 0 {
        return Err(Error::internal(
            "IPC bytes contain no data for query key".to_string(),
        ));
    }

    Ok(batches[0].column(0).clone())
}

pub fn query_to_proto(query: &Query) -> Result<pb::VectorQueryProto> {
    let key_arrow_ipc = array_to_ipc_bytes(query.key.as_ref())?;

    let metric_type = query.metric_type.map(|dt| dt.to_string());

    Ok(pb::VectorQueryProto {
        key_arrow_ipc,
        column: query.column.clone(),
        k: query.k as u32,
        lower_bound: query.lower_bound,
        upper_bound: query.upper_bound,
        minimum_nprobes: query.minimum_nprobes as u32,
        maximum_nprobes: query.maximum_nprobes.map(|n| n as u32),
        ef: query.ef.map(|n| n as u32),
        refine_factor: query.refine_factor,
        metric_type,
        use_index: query.use_index,
        dist_q_c: query.dist_q_c,
    })
}

pub fn query_from_proto(proto: pb::VectorQueryProto) -> Result<Query> {
    let key = array_from_ipc_bytes(&proto.key_arrow_ipc)?;

    let metric_type = proto
        .metric_type
        .as_deref()
        .map(DistanceType::try_from)
        .transpose()
        .map_err(|e| Error::internal(format!("Invalid distance type: {e}")))?;

    Ok(Query {
        column: proto.column,
        key,
        k: proto.k as usize,
        lower_bound: proto.lower_bound,
        upper_bound: proto.upper_bound,
        minimum_nprobes: proto.minimum_nprobes as usize,
        maximum_nprobes: proto.maximum_nprobes.map(|n| n as usize),
        ef: proto.ef.map(|n| n as usize),
        refine_factor: proto.refine_factor,
        metric_type,
        use_index: proto.use_index,
        dist_q_c: proto.dist_q_c,
    })
}

// =============================================================================
// ANNIvfPartitionExec <-> Proto
// =============================================================================

/// Convert an [`ANNIvfPartitionExec`] to proto for serialization.
pub async fn ann_ivf_partition_exec_to_proto(
    exec: &ANNIvfPartitionExec,
) -> Result<pb::AnnIvfPartitionExecProto> {
    let table = table_identifier_from_dataset(&exec.dataset).await?;
    let query = query_to_proto(&exec.query)?;

    Ok(pb::AnnIvfPartitionExecProto {
        query: Some(query),
        table: Some(table),
        index_uuids: exec.index_uuids.clone(),
    })
}

/// Reconstruct an [`ANNIvfPartitionExec`] from proto.
pub async fn ann_ivf_partition_exec_from_proto(
    proto: pb::AnnIvfPartitionExecProto,
    dataset: Option<Arc<Dataset>>,
) -> Result<ANNIvfPartitionExec> {
    let dataset = resolve_dataset(dataset, proto.table.as_ref()).await?;

    let query_proto = proto.query.ok_or_else(|| {
        Error::invalid_input_source("Missing VectorQueryProto in ANNIvfPartitionExecProto".into())
    })?;
    let query = query_from_proto(query_proto)?;

    ANNIvfPartitionExec::try_new(dataset, proto.index_uuids, query)
}

// =============================================================================
// ANNIvfSubIndexExec <-> Proto
// =============================================================================

/// Convert an [`ANNIvfSubIndexExec`] to proto for serialization.
pub async fn ann_ivf_sub_index_exec_to_proto(
    exec: &ANNIvfSubIndexExec,
) -> Result<pb::AnnIvfSubIndexExecProto> {
    let table = table_identifier_from_dataset(exec.dataset()).await?;
    let query = query_to_proto(exec.query())?;

    let indices: Vec<Vec<u8>> = exec
        .indices()
        .iter()
        .map(|idx| {
            let idx_proto: table_pb::IndexMetadata = idx.into();
            idx_proto.encode_to_vec()
        })
        .collect();

    Ok(pb::AnnIvfSubIndexExecProto {
        query: Some(query),
        table: Some(table),
        indices,
    })
}

/// Reconstruct an [`ANNIvfSubIndexExec`] from proto.
///
/// The caller (codec) is responsible for extracting child inputs:
/// - `input`: the child ANNIvfPartitionExec
/// - `prefilter_source`: optional prefilter input
pub async fn ann_ivf_sub_index_exec_from_proto(
    proto: pb::AnnIvfSubIndexExecProto,
    dataset: Option<Arc<Dataset>>,
    input: Arc<dyn datafusion::physical_plan::ExecutionPlan>,
    prefilter_source: PreFilterSource,
) -> Result<ANNIvfSubIndexExec> {
    let dataset = resolve_dataset(dataset, proto.table.as_ref()).await?;

    let query_proto = proto.query.ok_or_else(|| {
        Error::invalid_input_source("Missing VectorQueryProto in ANNIvfSubIndexExecProto".into())
    })?;
    let query = query_from_proto(query_proto)?;

    let indices: Vec<IndexMetadata> = proto
        .indices
        .iter()
        .map(|bytes| {
            let idx_proto = table_pb::IndexMetadata::decode(bytes.as_slice())
                .map_err(|e| Error::internal(format!("Failed to decode IndexMetadata: {e}")))?;
            IndexMetadata::try_from(idx_proto)
        })
        .collect::<Result<Vec<_>>>()?;

    if indices.is_empty() {
        return Err(Error::invalid_input_source(
            "ANNIvfSubIndexExecProto contains no indices".into(),
        ));
    }

    ANNIvfSubIndexExec::try_new(input, dataset, indices, query, prefilter_source)
}

// =============================================================================
// Helpers
// =============================================================================

/// Resolve a dataset from an optional pre-loaded instance or from the table identifier.
async fn resolve_dataset(
    dataset: Option<Arc<Dataset>>,
    table_id: Option<&pb::TableIdentifier>,
) -> Result<Arc<Dataset>> {
    match dataset {
        Some(ds) => Ok(ds),
        None => {
            let table_id = table_id.ok_or_else(|| {
                Error::invalid_input_source("Missing TableIdentifier in proto".into())
            })?;
            open_dataset_from_table_identifier(table_id).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::types::{Float32Type, UInt32Type};
    use arrow_array::{ArrayRef, Float32Array, Float64Array};
    use half::f16;
    use lance_datagen::{array, gen_batch};

    use crate::index::DatasetIndexExt;
    use crate::index::vector::VectorIndexParams;
    use lance_index::IndexType;
    use lance_index::vector::ivf::IvfBuildParams;
    use lance_index::vector::pq::PQBuildParams;

    #[test]
    fn test_array_ipc_roundtrip_f32() {
        let arr: ArrayRef = Arc::new(Float32Array::from(vec![1.0, 2.0, 3.0]));
        let bytes = array_to_ipc_bytes(arr.as_ref()).unwrap();
        let back = array_from_ipc_bytes(&bytes).unwrap();
        assert_eq!(arr.data_type(), back.data_type());
        assert_eq!(arr.len(), back.len());
    }

    #[test]
    fn test_array_ipc_roundtrip_f64() {
        let arr: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0]));
        let bytes = array_to_ipc_bytes(arr.as_ref()).unwrap();
        let back = array_from_ipc_bytes(&bytes).unwrap();
        assert_eq!(arr.data_type(), back.data_type());
        assert_eq!(&*arr, &*back);
    }

    #[test]
    fn test_array_ipc_roundtrip_f16() {
        let arr: ArrayRef = Arc::new(arrow_array::Float16Array::from(vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
        ]));
        let bytes = array_to_ipc_bytes(arr.as_ref()).unwrap();
        let back = array_from_ipc_bytes(&bytes).unwrap();
        assert_eq!(arr.data_type(), back.data_type());
        assert_eq!(arr.len(), back.len());
    }

    #[test]
    fn test_query_roundtrip() {
        let key: ArrayRef = Arc::new(Float32Array::from(vec![0.1, 0.2, 0.3]));
        let query = Query {
            column: "vector".to_string(),
            key,
            k: 10,
            lower_bound: Some(0.5),
            upper_bound: Some(1.5),
            minimum_nprobes: 4,
            maximum_nprobes: Some(16),
            ef: Some(64),
            refine_factor: Some(2),
            metric_type: Some(DistanceType::Cosine),
            use_index: true,
            dist_q_c: 0.42,
        };

        let proto = query_to_proto(&query).unwrap();
        let back = query_from_proto(proto).unwrap();

        assert_eq!(query.column, back.column);
        assert_eq!(query.k, back.k);
        assert_eq!(query.lower_bound, back.lower_bound);
        assert_eq!(query.upper_bound, back.upper_bound);
        assert_eq!(query.minimum_nprobes, back.minimum_nprobes);
        assert_eq!(query.maximum_nprobes, back.maximum_nprobes);
        assert_eq!(query.ef, back.ef);
        assert_eq!(query.refine_factor, back.refine_factor);
        assert_eq!(query.metric_type, back.metric_type);
        assert_eq!(query.use_index, back.use_index);
        assert_eq!(query.dist_q_c, back.dist_q_c);
        assert_eq!(query.key.len(), back.key.len());
        assert_eq!(query.key.data_type(), back.key.data_type());
    }

    #[test]
    fn test_query_roundtrip_none_metric() {
        let key: ArrayRef = Arc::new(Float32Array::from(vec![1.0]));
        let query = Query {
            column: "v".to_string(),
            key,
            k: 5,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: 1,
            maximum_nprobes: None,
            ef: None,
            refine_factor: None,
            metric_type: None,
            use_index: false,
            dist_q_c: 0.0,
        };

        let proto = query_to_proto(&query).unwrap();
        let back = query_from_proto(proto).unwrap();
        assert!(back.metric_type.is_none());
        assert!(!back.use_index);
    }

    async fn make_vector_dataset() -> (Arc<Dataset>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let batch = gen_batch()
            .col("id", array::step::<UInt32Type>())
            .col(
                "vector",
                array::rand_vec::<Float32Type>(lance_datagen::Dimension::from(128)),
            )
            .into_batch_rows(lance_datagen::RowCount::from(256))
            .unwrap();
        let path = dir.path().join("test_ann.lance");
        let ds = Dataset::write(
            arrow_array::RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
            path.to_str().unwrap(),
            None,
        )
        .await
        .unwrap();
        (Arc::new(ds), dir)
    }

    async fn make_indexed_dataset() -> (Arc<Dataset>, tempfile::TempDir) {
        let (_dataset, dir) = make_vector_dataset().await;
        let mut ds = Dataset::open(dir.path().join("test_ann.lance").to_str().unwrap())
            .await
            .unwrap();

        let ivf_params = IvfBuildParams::new(2);
        let pq_params = PQBuildParams::default();
        let index_params =
            VectorIndexParams::with_ivf_pq_params(DistanceType::L2, ivf_params, pq_params);

        ds.create_index(&["vector"], IndexType::Vector, None, &index_params, false)
            .await
            .unwrap();
        let ds = Dataset::open(dir.path().join("test_ann.lance").to_str().unwrap())
            .await
            .unwrap();
        (Arc::new(ds), dir)
    }

    #[tokio::test]
    async fn test_ann_ivf_partition_proto_roundtrip() {
        let (dataset, _dir) = make_vector_dataset().await;

        let key: ArrayRef = Arc::new(Float32Array::from(vec![0.1f32; 128]));
        let query = Query {
            column: "vector".to_string(),
            key,
            k: 10,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: 4,
            maximum_nprobes: Some(20),
            ef: None,
            refine_factor: None,
            metric_type: Some(DistanceType::L2),
            use_index: true,
            dist_q_c: 0.0,
        };

        let exec =
            ANNIvfPartitionExec::try_new(dataset.clone(), vec!["uuid-1".into()], query).unwrap();

        let proto = ann_ivf_partition_exec_to_proto(&exec).await.unwrap();

        // Check table identifier
        let table = proto.table.as_ref().unwrap();
        assert_eq!(table.uri, dataset.uri());
        assert_eq!(table.version, dataset.manifest.version);

        // Roundtrip
        let back = ann_ivf_partition_exec_from_proto(proto, Some(dataset.clone()))
            .await
            .unwrap();
        assert_eq!(back.query.column, "vector");
        assert_eq!(back.query.k, 10);
        assert_eq!(back.query.minimum_nprobes, 4);
        assert_eq!(back.query.maximum_nprobes, Some(20));
        assert_eq!(back.index_uuids, vec!["uuid-1".to_string()]);
    }

    #[tokio::test]
    async fn test_ann_ivf_sub_index_proto_roundtrip() {
        let (dataset, _dir) = make_indexed_dataset().await;

        // Get real index metadata from the dataset
        let indices = dataset.load_indices_by_name("vector_idx").await.unwrap();
        assert!(!indices.is_empty());

        let key: ArrayRef = Arc::new(Float32Array::from(vec![0.1f32; 128]));
        let query = Query {
            column: "vector".to_string(),
            key,
            k: 10,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: 2,
            maximum_nprobes: Some(4),
            ef: None,
            refine_factor: Some(2),
            metric_type: Some(DistanceType::L2),
            use_index: true,
            dist_q_c: 0.0,
        };

        // Build the partition exec as input child
        let partition_exec = ANNIvfPartitionExec::try_new(
            dataset.clone(),
            indices.iter().map(|idx| idx.uuid.to_string()).collect(),
            query.clone(),
        )
        .unwrap();

        let exec = ANNIvfSubIndexExec::try_new(
            Arc::new(partition_exec),
            dataset.clone(),
            indices.clone(),
            query,
            PreFilterSource::None,
        )
        .unwrap();

        // Encode
        let proto = ann_ivf_sub_index_exec_to_proto(&exec).await.unwrap();
        assert_eq!(proto.indices.len(), indices.len());

        // Decode — need a partition exec as input child
        let input_query = query_from_proto(proto.query.clone().unwrap()).unwrap();
        let input_partition = ANNIvfPartitionExec::try_new(
            dataset.clone(),
            indices.iter().map(|idx| idx.uuid.to_string()).collect(),
            input_query,
        )
        .unwrap();

        let back = ann_ivf_sub_index_exec_from_proto(
            proto,
            Some(dataset.clone()),
            Arc::new(input_partition),
            PreFilterSource::None,
        )
        .await
        .unwrap();

        assert_eq!(back.query().column, "vector");
        assert_eq!(back.query().k, 10);
        assert_eq!(back.query().minimum_nprobes, 2);
        assert_eq!(back.query().refine_factor, Some(2));
        assert_eq!(back.indices().len(), indices.len());
        for (original, decoded) in indices.iter().zip(back.indices().iter()) {
            assert_eq!(original.uuid, decoded.uuid);
            assert_eq!(original.name, decoded.name);
            assert_eq!(original.dataset_version, decoded.dataset_version);
            assert_eq!(original.fields, decoded.fields);
        }
    }
}
