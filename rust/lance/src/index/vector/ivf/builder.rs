// Copyright 2024 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use lance_file::writer::FileWriter;
use lance_table::io::manifest::ManifestDescribing;
use object_store::path::Path;
use snafu::{location, Location};
use tracing::instrument;

use lance_core::{Error, Result, ROW_ID};
use lance_index::vector::{
    hnsw::{builder::HnswBuildParams, HnswMetadata},
    ivf::shuffler::shuffle_dataset,
    pq::ProductQuantizer,
};
use lance_io::{stream::RecordBatchStream, traits::Writer};
use lance_linalg::distance::MetricType;

use crate::{
    index::vector::ivf::{io::write_pq_partitions, Ivf},
    Dataset,
};

use super::io::write_hnsw_index_partitions;

/// Build specific partitions of IVF index.
///
///
#[allow(clippy::too_many_arguments)]
#[instrument(level = "debug", skip(writer, data, ivf, pq))]
pub(super) async fn build_partitions(
    writer: &mut dyn Writer,
    data: impl RecordBatchStream + Unpin + 'static,
    column: &str,
    ivf: &mut Ivf,
    pq: Arc<dyn ProductQuantizer>,
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

    let ivf_model = lance_index::vector::ivf::new_ivf_with_pq(
        ivf.centroids.values(),
        ivf.centroids.value_length() as usize,
        metric_type,
        column,
        pq.clone(),
        Some(part_range),
    )?;

    let stream = shuffle_dataset(
        data,
        column,
        ivf_model,
        precomputed_partitons,
        ivf.num_partitions() as u32,
        pq.num_sub_vectors(),
        shuffle_partition_batches,
        shuffle_partition_concurrency,
        precomputed_shuffle_buffers,
    )
    .await?;

    write_pq_partitions(writer, ivf, Some(stream), None).await?;

    Ok(())
}

/// Build specific partitions of IVF index.
///
///
#[allow(clippy::too_many_arguments)]
#[instrument(level = "debug", skip(data, ivf, pq))]
pub(super) async fn build_hnsw_partitions(
    dataset: &Dataset,
    writer: &mut FileWriter<ManifestDescribing>,
    auxiliary_writer: Option<&mut FileWriter<ManifestDescribing>>,
    data: impl RecordBatchStream + Unpin + 'static,
    column: &str,
    ivf: &mut Ivf,
    pq: Arc<dyn ProductQuantizer>,
    metric_type: MetricType,
    hnsw_params: &HnswBuildParams,
    part_range: Range<u32>,
    precomputed_partitons: Option<HashMap<u64, u32>>,
    shuffle_partition_batches: usize,
    shuffle_partition_concurrency: usize,
    precomputed_shuffle_buffers: Option<(Path, Vec<String>)>,
) -> Result<Vec<HnswMetadata>> {
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

    let ivf_model = lance_index::vector::ivf::new_ivf_with_pq(
        ivf.centroids.values(),
        ivf.centroids.value_length() as usize,
        metric_type,
        column,
        pq.clone(),
        Some(part_range),
    )?;

    let stream = shuffle_dataset(
        data,
        column,
        ivf_model,
        precomputed_partitons,
        ivf.num_partitions() as u32,
        pq.num_sub_vectors(),
        shuffle_partition_batches,
        shuffle_partition_concurrency,
        precomputed_shuffle_buffers,
    )
    .await?;

    write_hnsw_index_partitions(
        dataset,
        column,
        metric_type,
        hnsw_params,
        writer,
        auxiliary_writer,
        ivf,
        pq,
        Some(stream),
        None,
    )
    .await
}
