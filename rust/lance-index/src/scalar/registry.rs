// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

// Re-export from lance-index-core for backwards compatibility
pub use lance_index_core::scalar::registry::{
    BasicTrainer, ScalarIndexCacheKey, ScalarIndexLoad, ScalarIndexPlugin, TrainingRequest,
    VALUE_COLUMN_NAME, single_flight_open,
};
pub use lance_index_core::scalar::expression::ScalarQueryParser;
pub use lance_index_core::scalar::seed::{FragmentSeed, IndexSeedWriter};
// Re-export training types that were previously defined here
pub use crate::scalar::{TrainingCriteria, TrainingOrdering};

use crate::scalar::TrainingCriteria as TrCriteria;

/// A default training request impl for indexes that don't need any parameters
pub(crate) struct DefaultTrainingRequest {
    criteria: TrCriteria,
}

impl DefaultTrainingRequest {
    pub fn new(criteria: TrCriteria) -> Self {
        Self { criteria }
    }
}

impl TrainingRequest for DefaultTrainingRequest {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn criteria(&self) -> &TrCriteria {
        &self.criteria
    }
}
