// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Graph Query Engine
//!
//! This crate provides graph query capabilities for Lance datasets using Cypher syntax.
//! It interprets Lance datasets as property graphs and translates Cypher queries into
//! DataFusion SQL queries for execution.
//!
//! # Features
//!
//! - Cypher query parsing and AST representation
//! - Graph pattern matching on columnar data
//! - Property graph interpretation of Lance datasets  
//! - Translation to optimized SQL via DataFusion
//! - Support for nodes, relationships, and properties
//!
//! # Example
//!
//! ```no_run
//! use lance_graph::{CypherQuery, GraphConfig, Result};
//!
//! # fn example() -> Result<()> {
//! let config = GraphConfig::builder()
//!     .with_node_label("Person", "person_id")
//!     .with_relationship("KNOWS", "src_person_id", "dst_person_id")
//!     .build()?;
//!
//! let query = CypherQuery::new("MATCH (p:Person) WHERE p.age > 30 RETURN p.name")?
//!     .with_config(config);
//!
//! // Execute against a dataset (would need actual dataset integration)
//! // let results = query.execute(&dataset).await?;
//! # Ok(())
//! # }
//! ```

pub mod ast;
pub mod config;
pub mod datafusion_planner;
pub mod error;
pub mod lance_native_planner;
pub mod logical_plan;
pub mod parser;
pub mod query;
pub mod query_processor;
pub mod semantic;
pub mod source_catalog;

pub use config::{GraphConfig, NodeMapping, RelationshipMapping};
pub use error::{GraphError, Result};
pub use query::CypherQuery;
