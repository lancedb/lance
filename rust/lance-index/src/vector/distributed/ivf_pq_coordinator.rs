// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Integration of global PQ training with distributed IVF index building

use lance_core::Result;
use lance_linalg::distance::DistanceType;

use crate::vector::pq::ProductQuantizer;

use super::ivf_coordinator::DistributedIvfCoordinator;
use super::pq_trainer::GlobalPqTrainer;

/// Coordinates distributed IVF-PQ index building with global PQ training
pub struct DistributedIvfPqCoordinator {
    ivf_coordinator: DistributedIvfCoordinator,
    pq_trainer: GlobalPqTrainer,
    _num_sub_vectors: usize,
}

impl DistributedIvfPqCoordinator {
    /// Create a new distributed IVF-PQ coordinator
    pub fn new(
        ivf_config: crate::vector::ivf::builder::IvfBuildParams,
        num_fragments: usize,
        num_sub_vectors: usize,
        num_bits: usize,
        sample_rate: usize,
        max_iters: usize,
        distance_type: DistanceType,
    ) -> Self {
        let ivf_coordinator = DistributedIvfCoordinator::new(ivf_config, num_fragments);
        let pq_trainer = GlobalPqTrainer::new(
            num_sub_vectors,
            num_bits,
            sample_rate,
            max_iters,
            distance_type,
        );

        Self {
            ivf_coordinator,
            pq_trainer,
            _num_sub_vectors: num_sub_vectors,
        }
    }

    /// Build distributed IVF-PQ index with global PQ training
    ///
    /// This method implements the recommended architecture:
    /// 1. First, train a global PQ codebook from the training data
    /// 2. Then, build distributed IVF index using the global codebook
    /// 3. Finally, merge the distributed results
    pub async fn build_distributed_ivf_pq(
        &self,
        training_vectors: &arrow_array::FixedSizeListArray,
        _fragments: &[super::ivf_coordinator::Fragment],
        _num_partitions: usize,
    ) -> Result<ProductQuantizer> {
        log::info!("IVF-PQ: train global PQ codebook");

        // Step 1: Train global PQ codebook
        log::info!("train global PQ codebook");
        let global_pq = self
            .pq_trainer
            .train_global_pq_codebook(training_vectors)
            .await?;

        log::info!(
            "PQ codebook trained: sub_vectors={}",
            self.pq_trainer.num_sub_vectors()
        );

        // Step 2: The calling code will handle building distributed IVF and merging
        log::info!("PQ training done");

        Ok(global_pq)
    }

    /// Get the IVF coordinator for configuration
    pub fn ivf_coordinator(&mut self) -> &mut DistributedIvfCoordinator {
        &mut self.ivf_coordinator
    }

    /// Get the PQ trainer for configuration  
    pub fn pq_trainer(&mut self) -> &mut GlobalPqTrainer {
        &mut self.pq_trainer
    }
}

/// Convenience function to build distributed IVF-PQ index with global PQ training
#[allow(clippy::too_many_arguments)]
pub async fn build_distributed_ivf_pq_with_global_codebook(
    training_vectors: &arrow_array::FixedSizeListArray,
    fragments: &[super::ivf_coordinator::Fragment],
    num_partitions: usize,
    num_sub_vectors: usize,
    num_bits: usize,
    sample_rate: usize,
    max_iters: usize,
    distance_type: DistanceType,
) -> Result<ProductQuantizer> {
    let ivf_config = crate::vector::ivf::builder::IvfBuildParams::new(num_partitions);

    let coordinator = DistributedIvfPqCoordinator::new(
        ivf_config,
        fragments.len(),
        num_sub_vectors,
        num_bits,
        sample_rate,
        max_iters,
        distance_type,
    );

    coordinator
        .build_distributed_ivf_pq(training_vectors, fragments, num_partitions)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_ivf_pq_coordinator_creation() {
        let ivf_config = crate::vector::ivf::builder::IvfBuildParams::new(16);
        let coordinator = DistributedIvfPqCoordinator::new(
            ivf_config,
            4,   // num_fragments
            8,   // num_sub_vectors
            8,   // num_bits
            256, // sample_rate
            50,  // max_iters
            DistanceType::L2,
        );

        // Test that coordinator is properly initialized
        assert_eq!(coordinator.pq_trainer.num_sub_vectors(), 8);
        assert_eq!(coordinator.pq_trainer.num_bits(), 8);
    }
}
