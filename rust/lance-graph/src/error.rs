// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Error types for the Lance graph query engine

use snafu::{prelude::*, Location};

pub type Result<T> = std::result::Result<T, GraphError>;

/// Errors that can occur during graph query processing
#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum GraphError {
    /// Error parsing Cypher query syntax
    #[snafu(display("Cypher parse error at position {position}: {message}"))]
    ParseError {
        message: String,
        position: usize,
        location: Location,
    },

    /// Error with graph configuration
    #[snafu(display("Graph configuration error: {message}"))]
    ConfigError { message: String, location: Location },

    /// Error during query planning
    #[snafu(display("Query planning error: {message}"))]
    PlanError { message: String, location: Location },

    /// Error during query execution
    #[snafu(display("Query execution error: {message}"))]
    ExecutionError { message: String, location: Location },

    /// Unsupported Cypher feature
    #[snafu(display("Unsupported Cypher feature: {feature}"))]
    UnsupportedFeature { feature: String, location: Location },

    /// Invalid graph pattern
    #[snafu(display("Invalid graph pattern: {message}"))]
    InvalidPattern { message: String, location: Location },

    /// DataFusion integration error
    #[snafu(display("DataFusion error: {source}"))]
    DataFusion {
        source: datafusion_common::DataFusionError,
        location: Location,
    },

    /// Lance core error
    #[snafu(display("Lance core error: {source}"))]
    LanceCore {
        source: lance_core::Error,
        location: Location,
    },

    /// Arrow error
    #[snafu(display("Arrow error: {source}"))]
    Arrow {
        source: arrow::error::ArrowError,
        location: Location,
    },
}

impl From<datafusion_common::DataFusionError> for GraphError {
    fn from(source: datafusion_common::DataFusionError) -> Self {
        Self::DataFusion {
            source,
            location: Location::new(file!(), line!(), column!()),
        }
    }
}

impl From<lance_core::Error> for GraphError {
    fn from(source: lance_core::Error) -> Self {
        Self::LanceCore {
            source,
            location: Location::new(file!(), line!(), column!()),
        }
    }
}

impl From<arrow::error::ArrowError> for GraphError {
    fn from(source: arrow::error::ArrowError) -> Self {
        Self::Arrow {
            source,
            location: Location::new(file!(), line!(), column!()),
        }
    }
}
