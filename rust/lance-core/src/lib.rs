// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_schema::{DataType, Field as ArrowField};

pub mod cache;
pub mod datatypes;
pub mod error;
pub mod utils;

pub use error::{Error, Result};

/// Column name for the meta row ID.
pub const ROW_ID: &str = "_rowid";
/// Column name for the meta row address.
pub const ROW_ADDR: &str = "_rowaddr";

lazy_static::lazy_static! {
    /// Row ID field. This is nullable because its validity bitmap is sometimes used
    /// as a selection vector.
    pub static ref ROW_ID_FIELD: ArrowField = ArrowField::new(ROW_ID, DataType::UInt64, true);
    /// Row address field. This is nullable because its validity bitmap is sometimes used
    /// as a selection vector.
    pub static ref ROW_ADDR_FIELD: ArrowField = ArrowField::new(ROW_ADDR, DataType::UInt64, true);
}
