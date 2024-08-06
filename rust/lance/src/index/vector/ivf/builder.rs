// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use arrow_array::FixedSizeListArray;
use futures::{StreamExt, TryStreamExt};
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu};
use lance_file::v2::writer::FileWriterOptions;
use lance_file::writer::FileWriter;
use lance_index::vector::quantizer::Quantizer;
use lance_index::vector::{ivf::storage::IvfModel, transform::Transformer};
use lance_io::object_writer::ObjectWriter;
use lance_table::io::manifest::ManifestDescribing;
use log::info;
use object_store::path::Path;
use snafu::{location, Location};
use tracing::instrument;

use lance_core::{traits::DatasetTakeRows, Error, Result, ROW_ID};
use lance_index::vector::{
    hnsw::{builder::HnswBuildParams, HnswMetadata},
    ivf::shuffler::shuffle_dataset,
    pq::ProductQuantizer,
};
use lance_io::{stream::RecordBatchStream, traits::Writer};
use lance_linalg::distance::{DistanceType, MetricType};

use crate::index::vector::ivf::io::write_pq_partitions;

use super::io::write_hnsw_quantization_index_partitions;

/// Build specific partitions of IVF index.
///
///
#[allow(clippy::too_many_arguments)]
#[instrument(level = "debug", skip_all)]
pub(super) async fn build_partitions(
    writer: &mut dyn Writer,
    data: impl RecordBatchStream + Unpin + 'static,
    column: &str,
    ivf: &mut IvfModel,
    pq: ProductQuantizer,
    metric_type: MetricType,
    part_range: Range<u32>,
    precomputed_partitons: Option<HashMap<u64, u32>>,
    shuffle_partition_batches: usize,
    shuffle_partition_concurrency: usize,
    precomputed_shuffle_buffers: Option<(Path, Vec<String>)>,
) -> Result<()> {
    let schema = data.schema();
    if schema.column_with_name(column).is_none() {
        return Err(Error::Schema {
            message: format!("column {} does not exist in data stream", column),
            location: location!(),
        });
    }
    if schema.column_with_name(ROW_ID).is_none() {
        return Err(Error::Schema {
            message: "ROW ID is not set when building index partitions".to_string(),
            location: location!(),
        });
    }

    println!("pq: {:?}", metric_type);
    let ivf_transformer = lance_index::vector::ivf::IvfTransformer::with_pq(
        ivf.centroids.clone().unwrap(),
        metric_type,
        column,
        pq.clone(),
        Some(part_range),
    );

    let stream = shuffle_dataset(
        data,
        column,
        ivf_transformer.into(),
        precomputed_partitons,
        ivf.num_partitions() as u32,
        shuffle_partition_batches,
        shuffle_partition_concurrency,
        precomputed_shuffle_buffers,
    )
    .await?;

    write_pq_partitions(writer, ivf, Some(stream), None).await?;

    Ok(())
}

pub async fn write_vector_storage(
    data: impl RecordBatchStream + Unpin + 'static,
    num_rows: u64,
    centroids: FixedSizeListArray,
    pq: ProductQuantizer,
    distance_type: DistanceType,
    column: &str,
    writer: ObjectWriter,
) -> Result<()> {
    info!("Transforming {} vectors for storage", num_rows);
    let ivf_transformer = Arc::new(lance_index::vector::ivf::IvfTransformer::with_pq(
        centroids,
        distance_type,
        column,
        pq,
        None,
    ));

    let mut writer =
        lance_file::v2::writer::FileWriter::new_lazy(writer, FileWriterOptions::default());
    let mut transformed_stream = data
        .map_ok(move |batch| {
            let ivf_transformer = ivf_transformer.clone();
            spawn_cpu(move || ivf_transformer.transform(&batch))
        })
        .try_buffer_unordered(get_num_compute_intensive_cpus());
    let mut total_rows_written = 0;
    while let Some(batch) = transformed_stream.next().await {
        let batch = batch?;
        total_rows_written += batch.num_rows();
        writer.write_batch(&batch).await?;
        info!("Transform progress: {}/{}", total_rows_written, num_rows);
    }
    writer.finish().await?;
    Ok(())
}

/// Build specific partitions of IVF index.
///
///
#[allow(clippy::too_many_arguments)]
#[instrument(level = "debug", skip(writer, auxiliary_writer, data, ivf, quantizer))]
pub(super) async fn build_hnsw_partitions(
    dataset: Arc<dyn DatasetTakeRows>,
    writer: &mut FileWriter<ManifestDescribing>,
    auxiliary_writer: Option<&mut FileWriter<ManifestDescribing>>,
    data: impl RecordBatchStream + Unpin + 'static,
    column: &str,
    ivf: &mut IvfModel,
    quantizer: Quantizer,
    metric_type: MetricType,
    hnsw_params: &HnswBuildParams,
    part_range: Range<u32>,
    precomputed_partitions: Option<HashMap<u64, u32>>,
    shuffle_partition_batches: usize,
    shuffle_partition_concurrency: usize,
    precomputed_shuffle_buffers: Option<(Path, Vec<String>)>,
) -> Result<(Vec<HnswMetadata>, IvfModel)> {
    let schema = data.schema();
    if schema.column_with_name(column).is_none() {
        return Err(Error::Schema {
            message: format!("column {} does not exist in data stream", column),
            location: location!(),
        });
    }
    if schema.column_with_name(ROW_ID).is_none() {
        return Err(Error::Schema {
            message: "ROW ID is not set when building index partitions".to_string(),
            location: location!(),
        });
    }

    let ivf_model = lance_index::vector::ivf::new_ivf_transformer_with_quantizer(
        ivf.centroids.clone().unwrap(),
        metric_type,
        column,
        quantizer.clone(),
        Some(part_range),
    )?;

    let stream = shuffle_dataset(
        data,
        column,
        ivf_model.into(),
        precomputed_partitions,
        ivf.num_partitions() as u32,
        shuffle_partition_batches,
        shuffle_partition_concurrency,
        precomputed_shuffle_buffers,
    )
    .await?;

    write_hnsw_quantization_index_partitions(
        dataset,
        column,
        metric_type,
        hnsw_params,
        writer,
        auxiliary_writer,
        ivf,
        quantizer,
        Some(stream),
        None,
    )
    .await
}
