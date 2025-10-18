// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_schema::{DataType, Field as ArrowField};
use std::sync::LazyLock;

pub mod cache;
pub mod container;
pub mod datatypes;
pub mod error;
pub mod traits;
pub mod utils;

pub use error::{box_error, ArrowResult, Error, Result};

/// Column name for the meta row ID.
pub const ROW_ID: &str = "_rowid";
/// Column name for the meta row address.
pub const ROW_ADDR: &str = "_rowaddr";
/// Column name for the meta row offset.
pub const ROW_OFFSET: &str = "_rowoffset";
/// Column name for the row's last updated at dataset version.
pub const ROW_LAST_UPDATED_AT_VERSION: &str = "_row_last_updated_at_version";
/// Column name for the row's created at dataset version.
pub const ROW_CREATED_AT_VERSION: &str = "_row_created_at_version";

/// Field ID for the meta row ID column. Uses i32::MAX to avoid conflicts with actual data fields.
pub const ROW_ID_FIELD_ID: i32 = i32::MAX;
/// Field ID for the meta row address column.
pub const ROW_ADDR_FIELD_ID: i32 = i32::MAX - 1;
/// Field ID for the meta row offset column.
pub const ROW_OFFSET_FIELD_ID: i32 = i32::MAX - 2;
/// Field ID for the row last updated at version column.
pub const ROW_LAST_UPDATED_AT_VERSION_FIELD_ID: i32 = i32::MAX - 3;
/// Field ID for the row created at version column.
pub const ROW_CREATED_AT_VERSION_FIELD_ID: i32 = i32::MAX - 4;

/// Row ID field. This is nullable because its validity bitmap is sometimes used
/// as a selection vector.
pub static ROW_ID_FIELD: LazyLock<ArrowField> =
    LazyLock::new(|| ArrowField::new(ROW_ID, DataType::UInt64, true));
/// Row address field. This is nullable because its validity bitmap is sometimes used
/// as a selection vector.
pub static ROW_ADDR_FIELD: LazyLock<ArrowField> =
    LazyLock::new(|| ArrowField::new(ROW_ADDR, DataType::UInt64, true));
/// Row offset field. This is nullable merely for compatibility with the other
/// fields.
pub static ROW_OFFSET_FIELD: LazyLock<ArrowField> =
    LazyLock::new(|| ArrowField::new(ROW_OFFSET, DataType::UInt64, true));
/// Row last updated at version field.
pub static ROW_LAST_UPDATED_AT_VERSION_FIELD: LazyLock<ArrowField> =
    LazyLock::new(|| ArrowField::new(ROW_LAST_UPDATED_AT_VERSION, DataType::UInt64, true));
/// Row created at version field.
pub static ROW_CREATED_AT_VERSION_FIELD: LazyLock<ArrowField> =
    LazyLock::new(|| ArrowField::new(ROW_CREATED_AT_VERSION, DataType::UInt64, true));
