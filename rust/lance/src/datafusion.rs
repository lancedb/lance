// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Extends DataFusion
//!

use datafusion::physical_plan::metrics::{Count, MetricValue, MetricsSet};

pub(crate) mod dataframe;
pub(crate) mod logical_plan;

pub trait MetricsExt {
    fn find_count(&self, name: &str) -> Option<Count>;
}

impl MetricsExt for MetricsSet {
    fn find_count(&self, metric_name: &str) -> Option<Count> {
        self.iter()
            .filter_map(|m| match m.value() {
                MetricValue::Count { name, count } => {
                    if name == metric_name {
                        Some(count.clone())
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .next()
    }
}

pub use dataframe::LanceTableProvider;
