// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Arrow Scalar - A scalar value representation for Apache Arrow types.
//!
//! This crate provides a `Scalar` enum for representing single Arrow values,
//! similar to DataFusion's `ScalarValue` but without DataFusion dependencies.
//!
//! # Features
//!
//! - Represents all Arrow primitive and complex types
//! - Converts between Arrow arrays and scalar values
//! - Byte serialization for storage and indexing
//! - Implements `Eq`, `Ord`, `Hash` with proper null/NaN handling
//!
//! # Example
//!
//! ```
//! use arrow_scalar::{Scalar, try_from_array};
//! use arrow_array::{Int32Array, Array};
//! use std::sync::Arc;
//!
//! // Create a scalar from an array element
//! let array = Int32Array::from(vec![Some(1), None, Some(3)]);
//! let scalar = try_from_array(&array, 0).unwrap();
//! assert_eq!(scalar, Scalar::Int32(Some(1)));
//!
//! // Convert back to an array
//! let arr = scalar.to_array();
//! assert_eq!(arr.len(), 1);
//! ```
//!
//! # Comparison Semantics (designed to match DataFusion's ScalarValue)
//!
//! - `NULL == NULL` for equality
//! - `NaN == NaN` using total_cmp semantics for floats
//! - Nulls sort first (less than all non-null values)
//! - Floats use `total_cmp()` for ordering

mod bytes;
mod cmp;
mod convert;
mod display;
mod scalar;

pub use convert::{iter_to_array, try_from_array};
pub use scalar::Scalar;
