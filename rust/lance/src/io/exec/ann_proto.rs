// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Protobuf serialization for [`ANNIvfPartitionExec`] and [`ANNIvfSubIndexExec`].
//!
//! Proto message definitions live in `crate::pb` (compiled from `ann.proto`).
//! Conversion functions live here because they need access to `ANNIvfSubIndexExec`
//! and `Dataset`, which are defined in this crate.
//!
//! A DataFusion `PhysicalExtensionCodec` can call these functions in `try_encode`
//! and `try_decode` to support distributed execution (planner → executor).

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{Field, Schema as ArrowSchema};
use lance_core::{Error, Result};
use lance_index::pb as index_pb;
use lance_index::vector::Query;
use lance_linalg::distance::DistanceType;
use lance_table::format::IndexMetadata;
use lance_table::format::pb as table_pb;

use crate::Dataset;
use crate::pb;

use super::knn::{ANNIvfPartitionExec, ANNIvfSubIndexExec};
use super::table_identifier::{resolve_dataset, table_identifier_from_dataset};
use super::utils::PreFilterSource;

// =============================================================================
// VectorQueryProto helpers
// =============================================================================

/// Serialize a query vector array to IPC file-format bytes.
///
/// Wraps the array in a single-column RecordBatch so that the IPC format
/// preserves the full data type (Float16, Float32, Float64, UInt8, etc.).
fn query_vector_to_ipc_bytes(array: &dyn arrow_array::Array) -> Result<Vec<u8>> {
    let field = Field::new("query_vector", array.data_type().clone(), true);
    let schema = Arc::new(ArrowSchema::new(vec![field]));
    let batch = RecordBatch::try_new(schema, vec![arrow_array::make_array(array.to_data())])
        .map_err(|e| {
            Error::internal(format!(
                "Failed to create RecordBatch for query vector: {e}"
            ))
        })?;

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

/// Deserialize a query vector array from IPC file-format bytes.
fn query_vector_from_ipc_bytes(bytes: &[u8]) -> Result<arrow_array::ArrayRef> {
    let cursor = std::io::Cursor::new(bytes);
    let reader = arrow_ipc::reader::FileReader::try_new(cursor, None)
        .map_err(|e| Error::internal(format!("Failed to create IPC reader: {e}")))?;

    let batches: Vec<RecordBatch> = reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| Error::internal(format!("Failed to read IPC batches: {e}")))?;

    if batches.is_empty() || batches[0].num_columns() == 0 {
        return Err(Error::internal(
            "IPC bytes contain no data for query vector".to_string(),
        ));
    }

    Ok(batches[0].column(0).clone())
}

pub fn query_to_proto(query: &Query) -> Result<pb::VectorQueryProto> {
    let query_vector_arrow_ipc = query_vector_to_ipc_bytes(query.key.as_ref())?;

    let metric_type = query
        .metric_type
        .map(|dt| index_pb::VectorMetricType::from(dt) as i32);

    Ok(pb::VectorQueryProto {
        query_vector_arrow_ipc,
        column: query.column.clone(),
        k: query.k as u32,
        lower_bound: query.lower_bound,
        upper_bound: query.upper_bound,
        minimum_nprobes: Some(query.minimum_nprobes as u32),
        maximum_nprobes: query.maximum_nprobes.map(|n| n as u32),
        ef: query.ef.map(|n| n as u32),
        refine_factor: query.refine_factor,
        metric_type,
        use_index: query.use_index,
        dist_q_c: Some(query.dist_q_c),
    })
}

pub fn query_from_proto(proto: pb::VectorQueryProto) -> Result<Query> {
    let key = query_vector_from_ipc_bytes(&proto.query_vector_arrow_ipc)?;

    let metric_type = proto
        .metric_type
        .map(|v| {
            index_pb::VectorMetricType::try_from(v)
                .map(DistanceType::from)
                .map_err(|_| Error::internal(format!("Invalid VectorMetricType value: {v}")))
        })
        .transpose()?;

    Ok(Query {
        column: proto.column,
        key,
        k: proto.k as usize,
        lower_bound: proto.lower_bound,
        upper_bound: proto.upper_bound,
        minimum_nprobes: proto.minimum_nprobes.unwrap_or(1) as usize,
        maximum_nprobes: proto.maximum_nprobes.map(|n| n as usize),
        ef: proto.ef.map(|n| n as usize),
        refine_factor: proto.refine_factor,
        metric_type,
        use_index: proto.use_index,
        dist_q_c: proto.dist_q_c.unwrap_or(0.0),
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

    if proto.index_uuids.is_empty() {
        return Err(Error::invalid_input_source(
            "ANNIvfPartitionExecProto contains no index UUIDs".into(),
        ));
    }

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

    let indices: Vec<table_pb::IndexMetadata> =
        exec.indices().iter().map(|idx| idx.into()).collect();

    Ok(pb::AnnIvfSubIndexExecProto {
        query: Some(query),
        table: Some(table),
        indices,
    })
}

/// Reconstruct an [`ANNIvfSubIndexExec`] from proto.
///
/// The caller (codec) is responsible for extracting child inputs:
/// - `input`: the child execution plan (e.g. partition exec)
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
        .into_iter()
        .map(IndexMetadata::try_from)
        .collect::<Result<Vec<_>>>()?;

    if indices.is_empty() {
        return Err(Error::invalid_input_source(
            "ANNIvfSubIndexExecProto contains no indices".into(),
        ));
    }

    ANNIvfSubIndexExec::try_new(input, dataset, indices, query, prefilter_source)
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
    use datafusion_physical_plan::test::TestMemoryExec;
    use lance_index::IndexType;
    use lance_index::vector::ivf::IvfBuildParams;
    use lance_index::vector::pq::PQBuildParams;

    #[test]
    fn test_query_vector_ipc_roundtrip_f32() {
        let arr: ArrayRef = Arc::new(Float32Array::from(vec![1.0, 2.0, 3.0]));
        let bytes = query_vector_to_ipc_bytes(arr.as_ref()).unwrap();
        let back = query_vector_from_ipc_bytes(&bytes).unwrap();
        assert_eq!(arr.data_type(), back.data_type());
        assert_eq!(arr.len(), back.len());
    }

    #[test]
    fn test_query_vector_ipc_roundtrip_f64() {
        let arr: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0]));
        let bytes = query_vector_to_ipc_bytes(arr.as_ref()).unwrap();
        let back = query_vector_from_ipc_bytes(&bytes).unwrap();
        assert_eq!(arr.data_type(), back.data_type());
        assert_eq!(&*arr, &*back);
    }

    #[test]
    fn test_query_vector_ipc_roundtrip_f16() {
        let arr: ArrayRef = Arc::new(arrow_array::Float16Array::from(vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
        ]));
        let bytes = query_vector_to_ipc_bytes(arr.as_ref()).unwrap();
        let back = query_vector_from_ipc_bytes(&bytes).unwrap();
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
        let (dataset, _dir) = make_indexed_dataset().await;

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

        let exec = ANNIvfPartitionExec::try_new(
            dataset.clone(),
            indices.iter().map(|idx| idx.uuid.to_string()).collect(),
            query,
        )
        .unwrap();

        let proto = ann_ivf_partition_exec_to_proto(&exec).await.unwrap();
        assert_eq!(proto.index_uuids.len(), indices.len());

        let back = ann_ivf_partition_exec_from_proto(proto, Some(dataset.clone()))
            .await
            .unwrap();

        assert_eq!(back.query.column, "vector");
        assert_eq!(back.query.k, 10);
        assert_eq!(back.query.minimum_nprobes, 2);
        assert_eq!(back.query.refine_factor, Some(2));
        assert_eq!(back.index_uuids.len(), indices.len());
        assert_eq!(back.dataset.uri(), dataset.uri());
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

        // Use a TestMemoryExec as a mock input child (provides the KNN_PARTITION_SCHEMA)
        let input: Arc<dyn datafusion::physical_plan::ExecutionPlan> =
            TestMemoryExec::try_new_exec(
                &[],
                super::super::knn::KNN_PARTITION_SCHEMA.clone(),
                None,
            )
            .unwrap();

        let exec = ANNIvfSubIndexExec::try_new(
            input.clone(),
            dataset.clone(),
            indices.clone(),
            query,
            PreFilterSource::None,
        )
        .unwrap();

        // Encode
        let proto = ann_ivf_sub_index_exec_to_proto(&exec).await.unwrap();
        assert_eq!(proto.indices.len(), indices.len());

        // Decode
        let back = ann_ivf_sub_index_exec_from_proto(
            proto,
            Some(dataset.clone()),
            input,
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
