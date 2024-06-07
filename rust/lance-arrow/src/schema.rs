// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Extension to arrow schema

use arrow_schema::{ArrowError, Field, FieldRef, Schema};

/// Extends the functionality of [arrow_schema::Schema].
pub trait SchemaExt {
    /// Create a new [`Schema`] with one extra field.
    fn try_with_column(&self, field: Field) -> std::result::Result<Schema, ArrowError>;

    fn try_with_column_at(
        &self,
        index: usize,
        field: Field,
    ) -> std::result::Result<Schema, ArrowError>;

    fn field_names(&self) -> Vec<&String>;

    fn without_column(&self, column_name: &str) -> Schema;
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

    fn try_with_column_at(
        &self,
        index: usize,
        field: Field,
    ) -> std::result::Result<Schema, ArrowError> {
        if self.column_with_name(field.name()).is_some() {
            return Err(ArrowError::SchemaError(format!(
                "Failed to modify schema: Inserting column {} would create a duplicate column in schema: {:?}",
                field.name(),
                self
            )));
        };
        let mut fields: Vec<FieldRef> = self.fields().iter().cloned().collect();
        fields.insert(index, FieldRef::new(field));
        Ok(Self::new_with_metadata(fields, self.metadata.clone()))
    }

    fn without_column(&self, column_name: &str) -> Schema {
        let fields: Vec<FieldRef> = self
            .fields()
            .iter()
            .filter(|f| f.name() != column_name)
            .cloned()
            .collect();
        Self::new_with_metadata(fields, self.metadata.clone())
    }

    fn field_names(&self) -> Vec<&String> {
        self.fields().iter().map(|f| f.name()).collect()
    }
}
