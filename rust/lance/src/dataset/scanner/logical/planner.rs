// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Stage 4: dispatching each Lance logical node to its lowering.
//!
//! The dispatch itself is mechanical. Every decision was made by a rule already; by the time a
//! node reaches here it names exactly one execution strategy.
//!
//! A plain scan has no Lance node above its leaf — the leaf is a `TableProvider`, which
//! `DefaultPhysicalPlanner` lowers on its own — so there is nothing to dispatch yet. Each node
//! type registers itself here as it arrives.

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::execution::session_state::SessionState;
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNode};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{ExtensionPlanner, PhysicalPlanner};

#[derive(Debug, Default)]
pub struct LanceExtensionPlanner;

#[async_trait]
impl ExtensionPlanner for LanceExtensionPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        _node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        _physical_inputs: &[Arc<dyn ExecutionPlan>],
        _session_state: &SessionState,
    ) -> datafusion::common::Result<Option<Arc<dyn ExecutionPlan>>> {
        Ok(None)
    }
}
