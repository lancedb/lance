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

use arrow_schema::{DataType, Field as ArrowField};

pub mod cache;
pub mod datatypes;
pub mod encodings;
pub mod error;
pub mod format;
pub mod io;
pub mod utils;

pub use error::{Error, Result};

/// Column name for the meta row ID.
pub const ROW_ID: &str = "_rowid";

lazy_static::lazy_static! {
    /// Row ID field. This is nullable because its validity bitmap is sometimes used
    /// as a selection vector.
    pub static ref ROW_ID_FIELD: ArrowField = ArrowField::new(ROW_ID, DataType::UInt64, true);
}
