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

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, FixedSizeListArray};
use arrow_cast::CastOptions;
use arrow_schema::{ArrowError, DataType};

/// Customized [`arrow_cast::can_cast_types`] that handles cases not supported upstream yet.
pub fn can_cast_types(from_type: &DataType, to_type: &DataType) -> bool {
    use DataType::*;
    match (from_type, to_type) {
        // TODO: remove this once Arrow supports this upstream.
        // https://github.com/apache/arrow-rs/pull/5360
        (FixedSizeList(from_field, size_from), FixedSizeList(to_field, size_to)) => {
            size_from == size_to && can_cast_types(from_field.data_type(), to_field.data_type())
        }
        // TODO: support bfloat16 cast?
        _ => arrow_cast::can_cast_types(from_type, to_type),
    }
}

/// Customized [`arrow_cast::cast_with_options`] that handles cases not supported upstream yet.
pub fn cast_with_options(
    array: &dyn Array,
    to_type: &DataType,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError> {
    use DataType::*;
    match (array.data_type(), to_type) {
        (FixedSizeList(_, size_from), FixedSizeList(to_field, size_to)) if size_from == size_to => {
            let array = array.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
            let values = cast_with_options(array.values(), to_field.data_type(), cast_options)?;
            Ok(Arc::new(FixedSizeListArray::try_new(
                to_field.clone(),
                *size_from,
                values,
                array.nulls().cloned(),
            )?))
        }
        _ => arrow_cast::cast_with_options(array, to_type, cast_options),
    }
}
