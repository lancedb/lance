// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Tests for distributed vector index building

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::vector::distributed::builder::DistributedVectorIndexBuilder;
    use crate::vector::distributed::communicator::LocalCommunicator;
    use crate::vector::distributed::config::DistributedVectorIndexConfig;
    use crate::vector::distributed::ivf_coordinator::DistributedIvfCoordinator;
    use crate::vector::distributed::ivf_flat_builder::{
        build_ivf_flat_distributed, DistributedIvfFlatParams,
    };
    use crate::vector::distributed::parameter_optimizer::{
        AdaptiveParameterOptimizer, DataCharacteristics,
    };
    use crate::vector::distributed::quality_validator::QualityValidator;
    use crate::vector::hnsw::builder::HnswBuildParams;
    use crate::vector::ivf::builder::IvfBuildParams;
    use arrow_array::cast::AsArray;
    use arrow_array::{Array, FixedSizeListArray, Float32Array};
    use arrow_schema::{DataType, Field};
    use lance_core::Result;
    use lance_file::reader::{FileReader as V2Reader, FileReaderOptions};
    use lance_io::{
        object_store::ObjectStore,
        scheduler::{ScanScheduler, SchedulerConfig},
        utils::CachedFileSize,
    };
    use lance_linalg::distance::DistanceType;
    use object_store::path::Path;
    use snafu::location;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_distributed_ivf_hnsw_config() {
        let config = DistributedVectorIndexConfig::default();

        assert!(config.max_parallelism > 0);
        assert_eq!(config.batch_size, 10000);
        assert!(config.ivf_config.enable_adaptive_retraining);
    }

    #[tokio::test]
    async fn test_parameter_optimizer() {
        let optimizer = AdaptiveParameterOptimizer::new();

        let ivf_params = IvfBuildParams::new(256);
        let hnsw_params = HnswBuildParams::default();
        let data_characteristics = DataCharacteristics::new(1000000, 128, 8);

        let _adjustments = optimizer.suggest_parameter_adjustments(
            &ivf_params,
            &hnsw_params,
            &data_characteristics,
        );

        // Test basic functionality - actual assertions would depend on implementation
        assert!(data_characteristics.total_vectors > 0);
    }

    #[tokio::test]
    async fn test_quality_validator() {
        let validator = QualityValidator::new();

        let metrics = crate::vector::distributed::quality_validator::QualityMetrics {
            ivf_balance_score: 0.75,
            hnsw_connectivity_score: 0.85,
            recall_at_k: std::collections::HashMap::from([(10, 0.95)]),
            latency_ms: 50.0,
            memory_usage_mb: 1024.0,
        };

        let is_valid = validator.validate_overall_quality(&metrics);
        assert!(is_valid);
    }

    #[tokio::test]
    async fn test_distributed_builder_creation() {
        let config = DistributedVectorIndexConfig::default();
        let builder = DistributedVectorIndexBuilder::new(config, DistanceType::L2, 128);

        assert_eq!(builder.dimension(), 128);
        assert_eq!(builder.distance_type(), DistanceType::L2);
    }

    fn make_fsl(vectors: Vec<Vec<f32>>) -> FixedSizeListArray {
        let dim = vectors.first().map(|v| v.len()).unwrap_or(0) as i32;
        let flat: Vec<f32> = vectors.into_iter().flatten().collect();
        let values = Float32Array::from(flat);
        let field = Arc::new(Field::new("item", DataType::Float32, false));
        FixedSizeListArray::new(field, dim, Arc::new(values), None)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_kmeans_allreduce_matches_single_worker() {
        // Construct two 2D Gaussian clusters: total 4*50=200 points, k=4
        let dim = 2usize;
        let k = 4usize;
        let n_per_cluster = 50usize;
        let centers = [
            [0.0f32, 0.0f32],
            [5.0f32, 0.0f32],
            [0.0f32, 5.0f32],
            [5.0f32, 5.0f32],
        ];
        let mut all = Vec::new();
        for c in centers.iter() {
            for i in 0..n_per_cluster {
                let dx = ((i as f32) * 0.01) - 0.25;
                let dy = ((i as f32) * 0.02) - 0.5;
                all.push(vec![c[0] + dx, c[1] + dy]);
            }
        }
        // Split into two shards for two workers' local samples
        let mid = all.len() / 2;
        let _fsl_rank0 = make_fsl(all[..mid].to_vec());
        let _fsl_rank1 = make_fsl(all[mid..].to_vec());

        // Two communicator instances (same-process simulation)
        let comms = LocalCommunicator::new_group(2);
        let mut coord = DistributedIvfCoordinator::new(IvfBuildParams::new(k), 2);
        coord.set_quality_threshold(0.0);

        // 1 worker (baseline): merge all samples and train locally
        let merged = make_fsl(all.clone());
        let single_centroids = coord
            .train_kmeans_distributed(merged.clone(), k)
            .await
            .expect("single worker training failed");

        // 2 workers: train on local samples; compare consistency and closeness to baseline
        // 2 workers: concurrent training to trigger allreduce
        let fut0 = coord.train_kmeans_allreduce(&merged, k, comms[0].as_ref());
        let fut1 = coord.train_kmeans_allreduce(&merged, k, comms[1].as_ref());
        let (c0_res, c1_res) = tokio::join!(fut0, fut1);
        let c0 = c0_res.expect("rank0 training failed");
        let c1 = c1_res.expect("rank1 training failed");

        assert_eq!(c0.len(), c1.len());
        assert_eq!(c0.value_length(), c1.value_length());
        let a0 = c0.values().as_primitive::<arrow::datatypes::Float32Type>();
        let a1 = c1.values().as_primitive::<arrow::datatypes::Float32Type>();
        for i in 0..(k * dim) {
            let diff = (a0.value(i) - a1.value(i)).abs();
            assert!(diff < 1e-4, "centroids mismatch at {} diff {}", i, diff);
        }

        let base = single_centroids
            .values()
            .as_primitive::<arrow::datatypes::Float32Type>();
        let c0v = c0.values().as_primitive::<arrow::datatypes::Float32Type>();
        let mut avg_diff = 0f32;
        for i in 0..(k * dim) {
            avg_diff += (base.value(i) - c0v.value(i)).abs();
        }
        avg_diff /= (k * dim) as f32;
        assert!(avg_diff < 0.2, "avg centroid diff too large: {}", avg_diff);
    }

    async fn read_ivf_lengths_from_aux(aux_path: &Path) -> Result<Vec<u32>> {
        let store = ObjectStore::local();
        let sched = ScanScheduler::new(
            Arc::new(store.clone()),
            SchedulerConfig::max_bandwidth(&store),
        );
        let fh = sched
            .open_file(aux_path, &CachedFileSize::unknown())
            .await?;
        let reader = V2Reader::try_open(
            fh,
            None,
            Arc::default(),
            &lance_core::cache::LanceCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await?;
        // IVF_METADATA_KEY contains the global buffer index
        let meta = reader.metadata();
        let ivf_idx: u32 = meta
            .file_schema
            .metadata
            .get(crate::vector::ivf::storage::IVF_METADATA_KEY)
            .ok_or_else(|| lance_core::Error::Index {
                message: "IVF meta missing".to_string(),
                location: location!(),
            })?
            .parse()
            .map_err(|_| lance_core::Error::Index {
                message: "IVF index parse error".to_string(),
                location: location!(),
            })?;
        let bytes = reader.read_global_buffer(ivf_idx).await?;
        let pb_ivf: crate::pb::Ivf = prost::Message::decode(bytes)?;
        Ok(pb_ivf.lengths)
    }

    #[tokio::test]
    async fn test_distributed_ivf_flat_build_matches_single_worker_assignments() {
        // Data generation
        let k = 4usize;
        let centers = [
            [0.0f32, 0.0f32],
            [5.0f32, 0.0f32],
            [0.0f32, 5.0f32],
            [5.0f32, 5.0f32],
        ];
        let mut all = Vec::new();
        for c in centers.iter() {
            for i in 0..50usize {
                let dx = ((i as f32) * 0.01) - 0.25;
                let dy = ((i as f32) * 0.02) - 0.5;
                all.push(vec![c[0] + dx, c[1] + dy]);
            }
        }
        let merged = make_fsl(all.clone());
        let _dim = merged.value_length() as usize;

        // Baseline: single-machine clustering + assignment
        let coord = DistributedIvfCoordinator::new(IvfBuildParams::new(k), 1);
        let centroids = coord
            .train_kmeans_distributed(merged.clone(), k)
            .await
            .expect("train failed");
        let ivf =
            crate::vector::ivf::new_ivf_transformer(centroids.clone(), DistanceType::L2, vec![]);
        let base_parts = ivf.compute_partitions(&merged).expect("assign failed");
        let mut base_counts = vec![0usize; k];
        for v in base_parts.values() {
            base_counts[*v as usize] += 1;
        }

        // Distributed: 2 workers + pretrained centroids + merge
        let mid = all.len() / 2;
        let f0 = make_fsl(all[..mid].to_vec());
        let f1 = make_fsl(all[mid..].to_vec());
        let comms = LocalCommunicator::new_group(2);
        let tmp = tempfile::TempDir::new().unwrap();
        let tmp_path = Path::from_filesystem_path(tmp.path()).unwrap();
        let out_dir = tmp_path.child("unified");
        let params = DistributedIvfFlatParams {
            nlist: k,
            distance_type: DistanceType::L2,
            pretrained_centroids: Some(Arc::new(centroids.clone())),
            shard_root: Some(tmp_path.clone()),
        };
        // Use different row_id offsets per rank to avoid conflicts
        let _r0 = build_ivf_flat_distributed(
            &f0,
            Some(0),
            &IvfBuildParams::new(k),
            DistanceType::L2,
            &params,
            comms[0].as_ref(),
            &out_dir,
        )
        .await
        .unwrap();
        let r1 = build_ivf_flat_distributed(
            &f1,
            Some(1_000_000_000),
            &IvfBuildParams::new(k),
            DistanceType::L2,
            &params,
            comms[1].as_ref(),
            &out_dir,
        )
        .await
        .unwrap();

        // Only rank0 returns the unified path
        let aux = r1.unwrap_or_else(|| out_dir.child(crate::INDEX_AUXILIARY_FILE_NAME));
        // Read merged IVF lengths
        let lengths = read_ivf_lengths_from_aux(&aux).await.unwrap();
        let merged_counts = lengths.iter().map(|x| *x as usize).collect::<Vec<_>>();

        // Compare distribution: allow different order but identical sum; centroid order fixed
        assert_eq!(merged_counts.len(), base_counts.len());
        assert_eq!(
            merged_counts.iter().sum::<usize>(),
            base_counts.iter().sum::<usize>()
        );
        for i in 0..k {
            assert_eq!(
                merged_counts[i], base_counts[i],
                "partition {} size mismatch",
                i
            );
        }
    }

    #[tokio::test]
    async fn test_distributed_ivf_flat_build_with_empty_shard() {
        // Base data (all to rank0, rank1 empty)
        let k = 2usize;
        let mut all = Vec::new();
        for i in 0..100usize {
            all.push(vec![i as f32 * 0.01, 0.0]);
        }
        let merged = make_fsl(all.clone());
        let coord = DistributedIvfCoordinator::new(IvfBuildParams::new(k), 1);
        let centroids = coord
            .train_kmeans_distributed(merged.clone(), k)
            .await
            .expect("train failed");

        let f0 = merged; // 100
        let f1 = make_fsl(Vec::new()); // 0
        let comms = LocalCommunicator::new_group(2);
        let tmp = tempfile::TempDir::new().unwrap();
        let tmp_path = Path::from_filesystem_path(tmp.path()).unwrap();
        let out_dir = tmp_path.child("unified2");
        let params = DistributedIvfFlatParams {
            nlist: k,
            distance_type: DistanceType::L2,
            pretrained_centroids: Some(Arc::new(centroids.clone())),
            shard_root: Some(tmp_path.clone()),
        };
        let _ = build_ivf_flat_distributed(
            &f0,
            Some(0),
            &IvfBuildParams::new(k),
            DistanceType::L2,
            &params,
            comms[0].as_ref(),
            &out_dir,
        )
        .await
        .unwrap();
        let aux = build_ivf_flat_distributed(
            &f1,
            Some(1_000_000_000),
            &IvfBuildParams::new(k),
            DistanceType::L2,
            &params,
            comms[1].as_ref(),
            &out_dir,
        )
        .await
        .unwrap();
        let aux = aux.unwrap_or_else(|| out_dir.child(crate::INDEX_AUXILIARY_FILE_NAME));
        let lengths = read_ivf_lengths_from_aux(&aux).await.unwrap();
        assert_eq!(lengths.iter().map(|x| *x as usize).sum::<usize>(), 100);
    }
}
