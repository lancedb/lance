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

//! Extension to arrow schema

use arrow_schema::{ArrowError, Field, FieldRef, Schema};

/// Extends the functionality of [arrow_schema::Schema].
pub trait SchemaExt {
    /// Create a new [`Schema`] with one extra field.
    fn try_with_column(&self, field: Field) -> std::result::Result<Schema, ArrowError>;

    fn field_names(&self) -> Vec<&String>;
}

impl SchemaExt for Schema {
    fn try_with_column(&self, field: Field) -> std::result::Result<Schema, ArrowError> {
        if self.column_with_name(field.name()).is_some() {
            return Err(ArrowError::SchemaError(format!(
                "Can not append column {} on schema: {:?}",
                field.name(),
                self
            )));
        };
        let mut fields: Vec<FieldRef> = self.fields().iter().cloned().collect();
        fields.push(FieldRef::new(field));
        Ok(Self::new_with_metadata(fields, self.metadata.clone()))
    }

    fn field_names(&self) -> Vec<&String> {
        self.fields().iter().map(|f| f.name()).collect()
    }
}
