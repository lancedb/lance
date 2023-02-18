// Copyright 2023 Lance Developers.
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

//! Execution Plans

use std::sync::Arc;

use datafusion::{
    catalog::{
        catalog::{CatalogProvider, MemoryCatalogProvider},
        schema::{MemorySchemaProvider, SchemaProvider},
    },
    datasource::TableProvider,
    execution::{
        context::SessionState,
        runtime_env::{RuntimeConfig, RuntimeEnv},
    },
    prelude::SessionConfig,
};

mod knn;
mod planner;
mod scan;
mod take;

pub use knn::*;
pub use scan::LanceScanExec;
pub use take::GlobalTakeExec;

use crate::Result;
use crate::{dataset::Dataset, io::datafusion::LanceTableProvider};

/// Create a session state for a dataset, to serve the rest of I/Os.
///
pub(crate) fn create_session_state(dataset: Arc<Dataset>) -> Result<SessionState> {
    let session_config = SessionConfig::default();
    let runtime_config = RuntimeConfig::new();
    let runtime_env = Arc::new(RuntimeEnv::new(runtime_config)?);
    let session_state = SessionState::with_config_rt(session_config, runtime_env);

    let catelog = Arc::new(MemoryCatalogProvider::new());
    let schema_provider = Arc::new(MemorySchemaProvider::new());
    let table_provider: Arc<dyn TableProvider> = Arc::new(LanceTableProvider::new(dataset));
    schema_provider
        .as_ref()
        .register_table("t".to_string(), table_provider);
    catelog
        .as_ref()
        .register_schema("default", schema_provider)?;
    session_state
        .catalog_list()
        .register_catalog("lance".to_string(), catelog);
    println!(
        "Catalog: {:?}",
        session_state.catalog_list().catalog_names()
    );

    Ok(session_state)
}
