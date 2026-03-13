// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;

use lance::dataset::Dataset;
use lance::index::vector::VectorIndexParams;
use lance_core::{Error, Result};
use lance_index::scalar::{BuiltinIndexType, InvertedIndexParams, ScalarIndexParams};
use lance_index::vector::bq::RQBuildParams;
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_index::{DatasetIndexExt, IndexType};
use lance_namespace::models::{
    CreateTableIndexRequest, DescribeTableIndexStatsRequest, DescribeTableIndexStatsResponse,
    IndexContent, ListTableIndicesRequest, ListTableIndicesResponse,
};

pub(super) fn normalize_index_type(index_type: &str) -> String {
    // Accept common aliases used by upstream clients and examples.
    match index_type.trim().to_ascii_uppercase().as_str() {
        "FTS" => "INVERTED".to_string(),
        "LABEL_LIST" => "LABELLIST".to_string(),
        "BLOOM_FILTER" => "BLOOMFILTER".to_string(),
        other => other.to_string(),
    }
}

pub(super) fn parse_index_type(index_type: &str) -> Result<IndexType> {
    let normalized = normalize_index_type(index_type);
    // BloomFilter has a spelling difference between wire value and enum variant.
    if normalized == "BLOOMFILTER" {
        return Ok(IndexType::BloomFilter);
    }
    IndexType::try_from(normalized.as_str()).map_err(|e| {
        Error::invalid_input_source(format!("Invalid index_type '{}': {}", index_type, e).into())
    })
}

pub(super) fn validate_scalar_index_type(index_type: &str) -> Result<()> {
    let parsed = parse_index_type(index_type)?;
    if parsed.is_vector() {
        return Err(Error::invalid_input_source(
            format!(
                "create_table_scalar_index only supports scalar index types, got '{}'",
                index_type
            )
            .into(),
        ));
    }
    Ok(())
}

pub(super) fn as_i64(value: Option<&serde_json::Value>) -> Option<i64> {
    value.and_then(|v| {
        if let Some(x) = v.as_i64() {
            Some(x)
        } else {
            v.as_u64().and_then(|x| i64::try_from(x).ok())
        }
    })
}

pub(super) async fn create_table_index(
    dataset: &mut Dataset,
    request: &CreateTableIndexRequest,
) -> Result<()> {
    let index_type = parse_index_type(&request.index_type)?;
    let index_name = request.name.clone();
    let column = request.column.as_str();

    if index_type.is_vector() {
        // Vector index families share distance metric semantics and IVF partitioning.
        let metric = request.distance_type.as_deref().unwrap_or("l2");
        let ivf = IvfBuildParams::with_target_partition_size(index_type.target_partition_size());

        match index_type {
            IndexType::IvfFlat => {
                let params = VectorIndexParams::with_ivf_flat_params(
                    metric.try_into().map_err(|e| {
                        Error::invalid_input_source(
                            format!("Invalid distance_type '{}': {}", metric, e).into(),
                        )
                    })?,
                    ivf,
                );
                dataset
                    .create_index(&[column], index_type, index_name, &params, false)
                    .await?;
            }
            IndexType::IvfSq => {
                let params = VectorIndexParams::with_ivf_sq_params(
                    metric.try_into().map_err(|e| {
                        Error::invalid_input_source(
                            format!("Invalid distance_type '{}': {}", metric, e).into(),
                        )
                    })?,
                    ivf,
                    SQBuildParams::default(),
                );
                dataset
                    .create_index(&[column], index_type, index_name, &params, false)
                    .await?;
            }
            IndexType::IvfRq => {
                let params = VectorIndexParams::with_ivf_rq_params(
                    metric.try_into().map_err(|e| {
                        Error::invalid_input_source(
                            format!("Invalid distance_type '{}': {}", metric, e).into(),
                        )
                    })?,
                    ivf,
                    RQBuildParams::default(),
                );
                dataset
                    .create_index(&[column], index_type, index_name, &params, false)
                    .await?;
            }
            IndexType::IvfHnswFlat => {
                let params = VectorIndexParams::ivf_hnsw(
                    metric.try_into().map_err(|e| {
                        Error::invalid_input_source(
                            format!("Invalid distance_type '{}': {}", metric, e).into(),
                        )
                    })?,
                    ivf,
                    HnswBuildParams::default(),
                );
                dataset
                    .create_index(&[column], index_type, index_name, &params, false)
                    .await?;
            }
            IndexType::IvfHnswSq => {
                let params = VectorIndexParams::with_ivf_hnsw_sq_params(
                    metric.try_into().map_err(|e| {
                        Error::invalid_input_source(
                            format!("Invalid distance_type '{}': {}", metric, e).into(),
                        )
                    })?,
                    ivf,
                    HnswBuildParams::default(),
                    SQBuildParams::default(),
                );
                dataset
                    .create_index(&[column], index_type, index_name, &params, false)
                    .await?;
            }
            IndexType::IvfHnswPq => {
                let params = VectorIndexParams::with_ivf_hnsw_pq_params(
                    metric.try_into().map_err(|e| {
                        Error::invalid_input_source(
                            format!("Invalid distance_type '{}': {}", metric, e).into(),
                        )
                    })?,
                    ivf,
                    HnswBuildParams::default(),
                    PQBuildParams::default(),
                );
                dataset
                    .create_index(&[column], index_type, index_name, &params, false)
                    .await?;
            }
            IndexType::Vector | IndexType::IvfPq => {
                let params = VectorIndexParams::with_ivf_pq_params(
                    metric.try_into().map_err(|e| {
                        Error::invalid_input_source(
                            format!("Invalid distance_type '{}': {}", metric, e).into(),
                        )
                    })?,
                    ivf,
                    PQBuildParams::default(),
                );
                dataset
                    .create_index(&[column], index_type, index_name, &params, false)
                    .await?;
            }
            _ => {
                return Err(Error::invalid_input_source(
                    format!("Unsupported vector index_type '{}'", request.index_type).into(),
                ));
            }
        }
    } else if matches!(
        index_type,
        IndexType::BTree
            | IndexType::Bitmap
            | IndexType::LabelList
            | IndexType::NGram
            | IndexType::ZoneMap
            | IndexType::BloomFilter
    ) {
        // Scalar built-ins map directly to one builtin descriptor each.
        let builtin = match index_type {
            IndexType::BTree => BuiltinIndexType::BTree,
            IndexType::Bitmap => BuiltinIndexType::Bitmap,
            IndexType::LabelList => BuiltinIndexType::LabelList,
            IndexType::NGram => BuiltinIndexType::NGram,
            IndexType::ZoneMap => BuiltinIndexType::ZoneMap,
            IndexType::BloomFilter => BuiltinIndexType::BloomFilter,
            _ => unreachable!(),
        };
        let params = ScalarIndexParams::for_builtin(builtin);
        dataset
            .create_index(&[column], index_type, index_name, &params, false)
            .await?;
    } else if index_type == IndexType::Inverted {
        // Inverted index exposes several optional tokenizer/normalization knobs.
        let mut params = InvertedIndexParams::default();
        if let Some(with_position) = request.with_position {
            params = params.with_position(with_position);
        }
        if let Some(base_tokenizer) = request.base_tokenizer.clone() {
            params = params.base_tokenizer(base_tokenizer);
        }
        if let Some(language) = request.language.clone() {
            params = params.language(language.as_str()).map_err(|e| {
                Error::invalid_input_source(
                    format!("Invalid language '{}': {}", language, e).into(),
                )
            })?;
        }
        if let Some(max_token_length) = request.max_token_length {
            if max_token_length < 0 {
                return Err(Error::invalid_input_source(
                    format!(
                        "max_token_length must be non-negative, got {}",
                        max_token_length
                    )
                    .into(),
                ));
            }
            params = params.max_token_length(Some(max_token_length as usize));
        }
        if let Some(lower_case) = request.lower_case {
            params = params.lower_case(lower_case);
        }
        if let Some(stem) = request.stem {
            params = params.stem(stem);
        }
        if let Some(remove_stop_words) = request.remove_stop_words {
            params = params.remove_stop_words(remove_stop_words);
        }
        if let Some(ascii_folding) = request.ascii_folding {
            params = params.ascii_folding(ascii_folding);
        }

        dataset
            .create_index(&[column], index_type, index_name, &params, false)
            .await?;
    } else {
        return Err(Error::invalid_input_source(
            format!("Unsupported index_type '{}'", request.index_type).into(),
        ));
    }

    Ok(())
}

pub(super) async fn list_table_indices(
    dataset: &Dataset,
    request: &ListTableIndicesRequest,
) -> Result<ListTableIndicesResponse> {
    let schema = dataset.schema();
    let indices = dataset.load_indices().await?;

    // A dataset may keep multiple metadata versions per logical index name.
    // Surface only the latest metadata entry per name.
    let mut latest_by_name: HashMap<String, lance_table::format::IndexMetadata> = HashMap::new();
    for idx in indices.iter() {
        match latest_by_name.get(&idx.name) {
            Some(existing) => {
                let should_replace = idx.dataset_version > existing.dataset_version
                    || (idx.dataset_version == existing.dataset_version
                        && idx.created_at > existing.created_at);
                if should_replace {
                    latest_by_name.insert(idx.name.clone(), idx.clone());
                }
            }
            None => {
                latest_by_name.insert(idx.name.clone(), idx.clone());
            }
        }
    }

    let mut index_contents: Vec<IndexContent> = latest_by_name
        .into_values()
        .map(|idx| {
            // Convert field ids to current schema names; silently skip unknown ids.
            let columns = idx
                .fields
                .iter()
                .filter_map(|field_id| schema.field_by_id(*field_id).map(|f| f.name.clone()))
                .collect::<Vec<_>>();
            IndexContent {
                index_name: idx.name,
                index_uuid: idx.uuid.to_string(),
                columns,
                status: "done".to_string(),
            }
        })
        .collect();

    index_contents.sort_by(|a, b| a.index_name.cmp(&b.index_name));

    if let Some(start_after) = request.page_token.as_ref() {
        // Tokens use "start_after index_name" semantics to match list_* APIs.
        if let Some(index) = index_contents
            .iter()
            .position(|item| item.index_name.as_str() > start_after.as_str())
        {
            index_contents.drain(0..index);
        } else {
            index_contents.clear();
        }
    }

    let mut next_page_token = None;
    if let Some(limit) = request.limit {
        if limit <= 0 {
            return Err(Error::invalid_input_source(
                format!("limit must be positive, got {}", limit).into(),
            ));
        }
        let limit = limit as usize;
        if index_contents.len() > limit {
            // Return the last emitted name as continuation token.
            next_page_token = limit
                .checked_sub(1)
                .and_then(|idx| index_contents.get(idx))
                .map(|item| item.index_name.clone());
            index_contents.truncate(limit);
        }
    }

    Ok(ListTableIndicesResponse {
        indexes: index_contents,
        page_token: next_page_token,
    })
}

pub(super) async fn describe_table_index_stats(
    dataset: &Dataset,
    request: &DescribeTableIndexStatsRequest,
) -> Result<DescribeTableIndexStatsResponse> {
    let index_name = request
        .index_name
        .as_ref()
        .ok_or_else(|| Error::invalid_input_source("index_name is required".to_string().into()))?;

    let stats_str = dataset.index_statistics(index_name).await?;
    // Convert backend JSON stats payload into stable typed response fields.
    let stats_json: serde_json::Value = serde_json::from_str(&stats_str).map_err(|e| {
        Error::namespace_source(
            format!(
                "Failed to parse index statistics for '{}': {}",
                index_name, e
            )
            .into(),
        )
    })?;

    let num_indices = as_i64(stats_json.get("num_indices")).and_then(|v| i32::try_from(v).ok());
    Ok(DescribeTableIndexStatsResponse {
        distance_type: stats_json
            .get("distance_type")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        index_type: stats_json
            .get("index_type")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        num_indexed_rows: as_i64(stats_json.get("num_indexed_rows")),
        num_unindexed_rows: as_i64(stats_json.get("num_unindexed_rows")),
        num_indices,
    })
}
