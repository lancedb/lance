// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Schema

use std::{
    collections::{HashMap, HashSet},
    fmt::{self, Debug, Formatter},
};

use arrow_array::RecordBatch;
use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
use deepsize::DeepSizeOf;
use lance_arrow::*;
use snafu::{location, Location};

use super::field::{Field, SchemaCompareOptions, StorageClass};
use crate::{Error, Result};

/// Lance Schema.
#[derive(Default, Debug, Clone, DeepSizeOf)]
pub struct Schema {
    /// Top-level fields in the dataset.
    pub fields: Vec<Field>,
    /// Metadata of the schema
    pub metadata: HashMap<String, String>,
}

/// State for a pre-order DFS iterator over the fields of a schema.
struct SchemaFieldIterPreOrder<'a> {
    field_stack: Vec<&'a Field>,
}

impl<'a> SchemaFieldIterPreOrder<'a> {
    #[allow(dead_code)]
    fn new(schema: &'a Schema) -> Self {
        let mut field_stack = Vec::with_capacity(schema.fields.len() * 2);
        for field in schema.fields.iter().rev() {
            field_stack.push(field);
        }
        Self { field_stack }
    }
}

/// Iterator implementation for a pre-order traversal of fields
impl<'a> Iterator for SchemaFieldIterPreOrder<'a> {
    type Item = &'a Field;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(next_field) = self.field_stack.pop() {
            for child in next_field.children.iter().rev() {
                self.field_stack.push(child);
            }
            Some(next_field)
        } else {
            None
        }
    }
}

impl Schema {
    pub fn compare_with_options(&self, expected: &Self, options: &SchemaCompareOptions) -> bool {
        compare_fields(&self.fields, &expected.fields, options)
            && (!options.compare_metadata || self.metadata == expected.metadata)
    }

    pub fn explain_difference(
        &self,
        expected: &Self,
        options: &SchemaCompareOptions,
    ) -> Option<String> {
        let mut differences =
            explain_fields_difference(&self.fields, &expected.fields, options, None);

        if options.compare_metadata {
            if let Some(difference) =
                explain_metadata_difference(&self.metadata, &expected.metadata)
            {
                differences.push(difference);
            }
        }

        if differences.is_empty() {
            None
        } else {
            Some(differences.join(", "))
        }
    }

    pub fn retain_storage_class(&self, storage_class: StorageClass) -> Self {
        let fields = self
            .fields
            .iter()
            .filter(|f| f.storage_class() == storage_class)
            .cloned()
            .collect();
        Self {
            fields,
            metadata: self.metadata.clone(),
        }
    }

    /// Splits the schema into two schemas, one with default storage class fields and the other with blob storage class fields.
    /// If there are no blob storage class fields, the second schema will be `None`.
    /// The order of fields is preserved.
    pub fn partition_by_storage_class(&self) -> (Self, Option<Self>) {
        let mut local_fields = Vec::with_capacity(self.fields.len());
        let mut sibling_fields = Vec::with_capacity(self.fields.len());
        for field in self.fields.iter() {
            match field.storage_class() {
                StorageClass::Default => local_fields.push(field.clone()),
                StorageClass::Blob => sibling_fields.push(field.clone()),
            }
        }
        (
            Self {
                fields: local_fields,
                metadata: self.metadata.clone(),
            },
            if sibling_fields.is_empty() {
                None
            } else {
                Some(Self {
                    fields: sibling_fields,
                    metadata: self.metadata.clone(),
                })
            },
        )
    }

    pub fn has_dictionary_types(&self) -> bool {
        self.fields.iter().any(|f| f.has_dictionary_types())
    }

    pub fn check_compatible(&self, expected: &Self, options: &SchemaCompareOptions) -> Result<()> {
        if !self.compare_with_options(expected, options) {
            let difference = self.explain_difference(expected, options);
            Err(Error::SchemaMismatch {
                // unknown reason is messy but this shouldn't happen.
                difference: difference.unwrap_or("unknown reason".to_string()),
                location: location!(),
            })
        } else {
            Ok(())
        }
    }

    /// Convert to a compact string representation.
    ///
    /// This is intended for display purposes and not for serialization.
    pub fn to_compact_string(&self, indent: Indentation) -> String {
        ArrowSchema::from(self).to_compact_string(indent)
    }

    fn do_project<T: AsRef<str>>(&self, columns: &[T], err_on_missing: bool) -> Result<Self> {
        let mut candidates: Vec<Field> = vec![];
        for col in columns {
            let split = col.as_ref().split('.').collect::<Vec<_>>();
            let first = split[0];
            if let Some(field) = self.field(first) {
                let projected_field = field.project(&split[1..])?;
                if let Some(candidate_field) = candidates.iter_mut().find(|f| f.name == first) {
                    candidate_field.merge(&projected_field)?;
                } else {
                    candidates.push(projected_field)
                }
            } else if err_on_missing {
                return Err(Error::Schema {
                    message: format!("Column {} does not exist", col.as_ref()),
                    location: location!(),
                });
            }
        }

        Ok(Self {
            fields: candidates,
            metadata: self.metadata.clone(),
        })
    }

    /// Project the columns over the schema.
    ///
    /// ```ignore
    /// let schema = Schema::from(...);
    /// let projected = schema.project(&["col1", "col2.sub_col3.field4"])?;
    /// ```
    pub fn project<T: AsRef<str>>(&self, columns: &[T]) -> Result<Self> {
        self.do_project(columns, true)
    }

    /// Project the columns over the schema, dropping unrecognized columns
    pub fn project_or_drop<T: AsRef<str>>(&self, columns: &[T]) -> Result<Self> {
        self.do_project(columns, false)
    }

    /// Check that the top level fields don't contain `.` in their names
    /// to distinguish from nested fields.
    // TODO: pub(crate)
    pub fn validate(&self) -> Result<()> {
        let mut seen_names = HashSet::new();

        for field in self.fields.iter() {
            if field.name.contains('.') {
                return Err(Error::Schema{message:format!(
                    "Top level field {} cannot contain `.`. Maybe you meant to create a struct field?",
                    field.name.clone()
                ), location: location!(),});
            }

            let column_path = self
                .field_ancestry_by_id(field.id)
                .unwrap()
                .iter()
                .map(|f| f.name.as_str())
                .collect::<Vec<_>>()
                .join(".");
            if !seen_names.insert(column_path.clone()) {
                return Err(Error::Schema {
                    message: format!(
                        "Duplicate field name \"{}\" in schema:\n {:#?}",
                        column_path, self
                    ),
                    location: location!(),
                });
            }
        }

        // Check for duplicate field ids
        let mut seen_ids = HashSet::new();
        for field in self.fields_pre_order() {
            if field.id < 0 {
                return Err(Error::Schema {
                    message: format!("Field {} has a negative id {}", field.name, field.id),
                    location: location!(),
                });
            }
            if !seen_ids.insert(field.id) {
                return Err(Error::Schema {
                    message: format!("Duplicate field id {} in schema {:?}", field.id, self),
                    location: location!(),
                });
            }
        }

        Ok(())
    }

    /// Intersection between two [`Schema`].
    pub fn intersection(&self, other: &Self) -> Result<Self> {
        self.do_intersection(other, false)
    }

    /// Intersection between two [`Schema`], ignoring data types.
    pub fn intersection_ignore_types(&self, other: &Self) -> Result<Self> {
        self.do_intersection(other, true)
    }

    fn do_intersection(&self, other: &Self, ignore_types: bool) -> Result<Self> {
        let mut candidates: Vec<Field> = vec![];
        for field in other.fields.iter() {
            if let Some(candidate_field) = self.field(&field.name) {
                candidates.push(candidate_field.do_intersection(field, ignore_types)?);
            }
        }

        Ok(Self {
            fields: candidates,
            metadata: self.metadata.clone(),
        })
    }

    /// Iterates over the fields using a pre-order traversal
    ///
    /// This is a DFS traversal where the parent is visited
    /// before its children
    pub fn fields_pre_order(&self) -> impl Iterator<Item = &Field> {
        SchemaFieldIterPreOrder::new(self)
    }

    /// Returns a new schema that only contains the fields in `column_ids`.
    ///
    /// This projection can filter out both top-level and nested fields
    ///
    /// If `include_all_children` is true, then if a parent field id is passed,
    /// then all children of that field will be included in the projection
    /// regardless of whether their ids were passed. If this is false, then
    /// only the child fields with the passed ids will be included.
    pub fn project_by_ids(&self, column_ids: &[i32], include_all_children: bool) -> Self {
        let filtered_fields = self
            .fields
            .iter()
            .filter_map(|f| f.project_by_ids(column_ids, include_all_children))
            .collect();
        Self {
            fields: filtered_fields,
            metadata: self.metadata.clone(),
        }
    }

    /// Project the schema by another schema, and preserves field metadata, i.e., Field IDs.
    ///
    /// Parameters
    /// - `projection`: The schema to project by. Can be [`arrow_schema::Schema`] or [`Schema`].
    pub fn project_by_schema<S: TryInto<Self, Error = Error>>(
        &self,
        projection: S,
    ) -> Result<Self> {
        let projection = projection.try_into()?;
        let mut new_fields = vec![];
        for field in projection.fields.iter() {
            if let Some(self_field) = self.field(&field.name) {
                new_fields.push(self_field.project_by_field(field)?);
            } else {
                return Err(Error::Schema {
                    message: format!("Field {} not found", field.name),
                    location: location!(),
                });
            }
        }
        Ok(Self {
            fields: new_fields,
            metadata: self.metadata.clone(),
        })
    }

    /// Exclude the fields from `other` Schema, and returns a new Schema.
    pub fn exclude<T: TryInto<Self> + Debug>(&self, schema: T) -> Result<Self> {
        let other = schema.try_into().map_err(|_| Error::Schema {
            message: "The other schema is not compatible with this schema".to_string(),
            location: location!(),
        })?;
        let mut fields = vec![];
        for field in self.fields.iter() {
            if let Some(other_field) = other.field(&field.name) {
                if field.data_type().is_struct() {
                    if let Some(f) = field.exclude(other_field) {
                        fields.push(f)
                    }
                }
            } else {
                fields.push(field.clone());
            }
        }
        Ok(Self {
            fields,
            metadata: self.metadata.clone(),
        })
    }

    /// Get a field by name. Return `None` if the field does not exist.
    pub fn field(&self, name: &str) -> Option<&Field> {
        let split = name.split('.').collect::<Vec<_>>();
        self.fields
            .iter()
            .find(|f| f.name == split[0])
            .and_then(|c| c.sub_field(&split[1..]))
    }

    // TODO: This is not a public API, change to pub(crate) after refactor is done.
    pub fn field_id(&self, column: &str) -> Result<i32> {
        self.field(column)
            .map(|f| f.id)
            .ok_or_else(|| Error::Schema {
                message: "Vector column not in schema".to_string(),
                location: location!(),
            })
    }

    pub fn top_level_field_ids(&self) -> Vec<i32> {
        self.fields.iter().map(|f| f.id).collect()
    }

    // Recursively collect all the field IDs, in pre-order traversal order.
    // TODO: pub(crate)
    pub fn field_ids(&self) -> Vec<i32> {
        self.fields_pre_order().map(|f| f.id).collect()
    }

    /// Get field by its id.
    // TODO: pub(crate)
    pub fn field_by_id(&self, id: impl Into<i32>) -> Option<&Field> {
        let id = id.into();
        for field in self.fields.iter() {
            if field.id == id {
                return Some(field);
            }
            if let Some(grandchild) = field.field_by_id(id) {
                return Some(grandchild);
            }
        }
        None
    }

    /// Get the sequence of fields from the root to the field with the given id.
    pub fn field_ancestry_by_id(&self, id: i32) -> Option<Vec<&Field>> {
        let mut to_visit = self.fields.iter().map(|f| vec![f]).collect::<Vec<_>>();
        while let Some(path) = to_visit.pop() {
            let field = path.last().unwrap();
            if field.id == id {
                return Some(path);
            }
            for child in field.children.iter() {
                let mut new_path = path.clone();
                new_path.push(child);
                to_visit.push(new_path);
            }
        }
        None
    }

    pub fn mut_field_by_id(&mut self, id: impl Into<i32>) -> Option<&mut Field> {
        let id = id.into();
        for field in self.fields.as_mut_slice() {
            if field.id == id {
                return Some(field);
            }
            if let Some(grandchild) = field.mut_field_by_id(id) {
                return Some(grandchild);
            }
        }
        None
    }

    // TODO: pub(crate)
    /// Get the maximum field id in the schema.
    ///
    /// Note: When working with Datasets, you should prefer [Manifest::max_field_id()]
    /// over this method. This method does not take into account the field IDs
    /// of dropped fields.
    pub fn max_field_id(&self) -> Option<i32> {
        self.fields.iter().map(|f| f.max_id()).max()
    }

    /// Recursively attach set up dictionary values to the dictionary fields.
    // TODO: pub(crate)
    pub fn set_dictionary(&mut self, batch: &RecordBatch) -> Result<()> {
        for field in self.fields.as_mut_slice() {
            let column = batch
                .column_by_name(&field.name)
                .ok_or_else(|| Error::Schema {
                    message: format!("column '{}' does not exist in the record batch", field.name),
                    location: location!(),
                })?;
            field.set_dictionary(column);
        }
        Ok(())
    }

    /// Walk through the fields and assign a new field id to each field that does
    /// not have one (e.g. is set to -1)
    ///
    /// If this schema is on an existing dataset, pass the result of
    /// `Manifest::max_field_id` to `max_existing_id`. If for some reason that
    /// id is lower than the maximum field id in this schema, the field IDs will
    /// be reassigned starting from the maximum field id in this schema.
    ///
    /// If this schema is not associated with a dataset, pass `None` to
    /// `max_existing_id`. This is the same as passing [Self::max_field_id()].
    pub fn set_field_id(&mut self, max_existing_id: Option<i32>) {
        let schema_max_id = self.max_field_id().unwrap_or(-1);
        let max_existing_id = max_existing_id.unwrap_or(-1);
        let mut current_id = schema_max_id.max(max_existing_id) + 1;
        self.fields
            .iter_mut()
            .for_each(|f| f.set_id(-1, &mut current_id));
    }

    fn reset_id(&mut self) {
        self.fields.iter_mut().for_each(|f| f.reset_id());
    }

    /// Create a new schema by adding fields to the end of this schema
    pub fn extend(&mut self, fields: &[ArrowField]) -> Result<()> {
        let new_fields = fields
            .iter()
            .map(Field::try_from)
            .collect::<Result<Vec<_>>>()?;
        self.fields.extend(new_fields);
        // Validate this addition does not create any duplicate field names
        let field_names = self.fields.iter().map(|f| &f.name).collect::<HashSet<_>>();
        if field_names.len() != self.fields.len() {
            Err(Error::Internal {
                message: format!(
                    "Attempt to add fields [{:?}] would lead to duplicate field names",
                    fields.iter().map(|f| f.name()).collect::<Vec<_>>()
                ),
                location: location!(),
            })
        } else {
            Ok(())
        }
    }

    /// Merge this schema from the other schema.
    ///
    /// After merging, the field IDs from `other` schema will be reassigned,
    /// following the fields in `self`.
    pub fn merge<S: TryInto<Self, Error = Error>>(&self, other: S) -> Result<Self> {
        let mut other: Self = other.try_into()?;
        other.reset_id();

        let mut merged_fields: Vec<Field> = vec![];
        for mut field in self.fields.iter().cloned() {
            if let Some(other_field) = other.field(&field.name) {
                // if both are struct types, then merge the fields
                field.merge(other_field)?;
            }
            merged_fields.push(field);
        }

        // we already checked for overlap so just need to add new top-level fields
        // in the incoming schema
        for field in other.fields.as_slice() {
            if !merged_fields.iter().any(|f| f.name == field.name) {
                merged_fields.push(field.clone());
            }
        }
        let metadata = self
            .metadata
            .iter()
            .chain(other.metadata.iter())
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let schema = Self {
            fields: merged_fields,
            metadata,
        };
        Ok(schema)
    }
}

impl PartialEq for Schema {
    fn eq(&self, other: &Self) -> bool {
        self.fields == other.fields
    }
}

impl fmt::Display for Schema {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for field in self.fields.iter() {
            writeln!(f, "{field}")?
        }
        Ok(())
    }
}

/// Convert `arrow2::datatype::Schema` to Lance
impl TryFrom<&ArrowSchema> for Schema {
    type Error = Error;

    fn try_from(schema: &ArrowSchema) -> Result<Self> {
        let mut schema = Self {
            fields: schema
                .fields
                .iter()
                .map(|f| Field::try_from(f.as_ref()))
                .collect::<Result<_>>()?,
            metadata: schema.metadata.clone(),
        };
        schema.set_field_id(None);

        Ok(schema)
    }
}

/// Convert Lance Schema to Arrow Schema
impl From<&Schema> for ArrowSchema {
    fn from(schema: &Schema) -> Self {
        Self {
            fields: schema.fields.iter().map(ArrowField::from).collect(),
            metadata: schema.metadata.clone(),
        }
    }
}

/// Make API cleaner to accept both [`Schema`] and Arrow Schema.
impl TryFrom<&Self> for Schema {
    type Error = Error;

    fn try_from(schema: &Self) -> Result<Self> {
        Ok(schema.clone())
    }
}

pub fn compare_fields(
    fields: &[Field],
    expected: &[Field],
    options: &SchemaCompareOptions,
) -> bool {
    if options.allow_missing_if_nullable || options.ignore_field_order {
        let expected_names = expected
            .iter()
            .map(|f| f.name.as_str())
            .collect::<HashSet<_>>();
        for field in fields {
            if !expected_names.contains(field.name.as_str()) {
                // Extra field
                return false;
            }
        }

        let field_mapping = fields
            .iter()
            .enumerate()
            .map(|(pos, f)| (f.name.as_str(), (f, pos)))
            .collect::<HashMap<_, _>>();
        let mut cumulative_position = 0;
        for expected_field in expected {
            if let Some((field, pos)) = field_mapping.get(expected_field.name.as_str()) {
                if !field.compare_with_options(expected_field, options) {
                    return false;
                }
                if !options.ignore_field_order && *pos < cumulative_position {
                    return false;
                }
                cumulative_position = *pos;
            } else if options.allow_missing_if_nullable && expected_field.nullable {
                continue;
            } else {
                return false;
            }
        }
        true
    } else {
        // Fast path: we can just zip
        fields.len() == expected.len()
            && fields
                .iter()
                .zip(expected.iter())
                .all(|(lhs, rhs)| lhs.compare_with_options(rhs, options))
    }
}

pub fn explain_fields_difference(
    fields: &[Field],
    expected: &[Field],
    options: &SchemaCompareOptions,
    path: Option<&str>,
) -> Vec<String> {
    let field_names = fields
        .iter()
        .map(|f| f.name.as_str())
        .collect::<HashSet<_>>();
    let expected_names = expected
        .iter()
        .map(|f| f.name.as_str())
        .collect::<HashSet<_>>();

    let prepend_path = |f: &str| {
        if let Some(path) = path {
            format!("{}.{}", path, f)
        } else {
            f.to_string()
        }
    };

    // Check there are no extra fields or missing fields
    let unexpected_fields = field_names
        .difference(&expected_names)
        .cloned()
        .map(prepend_path)
        .collect::<Vec<_>>();
    let missing_fields = expected_names.difference(&field_names);
    let missing_fields = if options.allow_missing_if_nullable {
        missing_fields
            .filter(|f| {
                let expected_field = expected.iter().find(|ef| ef.name == **f).unwrap();
                !expected_field.nullable
            })
            .cloned()
            .map(prepend_path)
            .collect::<Vec<_>>()
    } else {
        missing_fields
            .cloned()
            .map(prepend_path)
            .collect::<Vec<_>>()
    };

    let mut differences = vec![];
    if !missing_fields.is_empty() || !unexpected_fields.is_empty() {
        differences.push(format!(
            "fields did not match, missing=[{}], unexpected=[{}]",
            missing_fields.join(", "),
            unexpected_fields.join(", ")
        ));
    }

    // Map the expected fields to position of field
    let field_mapping = expected
        .iter()
        .filter_map(|ef| {
            fields
                .iter()
                .position(|f| ef.name == f.name)
                .map(|pos| (ef, pos))
        })
        .collect::<Vec<_>>();

    // Check the fields are in the same order
    if !options.ignore_field_order {
        let fields_out_of_order = field_mapping.windows(2).any(|w| w[0].1 > w[1].1);
        if fields_out_of_order {
            let expected_order = expected.iter().map(|f| f.name.as_str()).collect::<Vec<_>>();
            let actual_order = fields.iter().map(|f| f.name.as_str()).collect::<Vec<_>>();
            differences.push(format!(
                "fields in different order, expected: [{}], actual: [{}]",
                expected_order.join(", "),
                actual_order.join(", ")
            ));
        }
    }

    // Check for individual differences in the fields
    for (expected_field, field_pos) in field_mapping.iter() {
        let field = &fields[*field_pos];
        debug_assert_eq!(field.name, expected_field.name);
        let field_diffs = field.explain_differences(expected_field, options, path);
        if !field_diffs.is_empty() {
            differences.push(field_diffs.join(", "))
        }
    }

    differences
}

fn explain_metadata_difference(
    metadata: &HashMap<String, String>,
    expected: &HashMap<String, String>,
) -> Option<String> {
    if metadata != expected {
        Some(format!(
            "metadata did not match, expected: {:?}, actual: {:?}",
            expected, metadata
        ))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    use arrow_schema::{
        DataType, Field as ArrowField, Fields as ArrowFields, Schema as ArrowSchema,
    };

    #[test]
    fn test_schema_projection() {
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new(
                "b",
                DataType::Struct(ArrowFields::from(vec![
                    ArrowField::new("f1", DataType::Utf8, true),
                    ArrowField::new("f2", DataType::Boolean, false),
                    ArrowField::new("f3", DataType::Float32, false),
                ])),
                true,
            ),
            ArrowField::new("c", DataType::Float64, false),
        ]);
        let schema = Schema::try_from(&arrow_schema).unwrap();
        let projected = schema.project(&["b.f1", "b.f3", "c"]).unwrap();

        let expected_arrow_schema = ArrowSchema::new(vec![
            ArrowField::new(
                "b",
                DataType::Struct(ArrowFields::from(vec![
                    ArrowField::new("f1", DataType::Utf8, true),
                    ArrowField::new("f3", DataType::Float32, false),
                ])),
                true,
            ),
            ArrowField::new("c", DataType::Float64, false),
        ]);
        assert_eq!(ArrowSchema::from(&projected), expected_arrow_schema);
    }

    #[test]
    fn test_schema_project_by_ids() {
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new(
                "b",
                DataType::Struct(ArrowFields::from(vec![
                    ArrowField::new("f1", DataType::Utf8, true),
                    ArrowField::new("f2", DataType::Boolean, false),
                    ArrowField::new("f3", DataType::Float32, false),
                ])),
                true,
            ),
            ArrowField::new("c", DataType::Float64, false),
        ]);
        let mut schema = Schema::try_from(&arrow_schema).unwrap();
        schema.set_field_id(None);
        let projected = schema.project_by_ids(&[2, 4, 5], true);

        let expected_arrow_schema = ArrowSchema::new(vec![
            ArrowField::new(
                "b",
                DataType::Struct(ArrowFields::from(vec![
                    ArrowField::new("f1", DataType::Utf8, true),
                    ArrowField::new("f3", DataType::Float32, false),
                ])),
                true,
            ),
            ArrowField::new("c", DataType::Float64, false),
        ]);
        assert_eq!(ArrowSchema::from(&projected), expected_arrow_schema);

        let projected = schema.project_by_ids(&[2], true);
        let expected_arrow_schema = ArrowSchema::new(vec![ArrowField::new(
            "b",
            DataType::Struct(ArrowFields::from(vec![ArrowField::new(
                "f1",
                DataType::Utf8,
                true,
            )])),
            true,
        )]);
        assert_eq!(ArrowSchema::from(&projected), expected_arrow_schema);

        let projected = schema.project_by_ids(&[1], true);
        let expected_arrow_schema = ArrowSchema::new(vec![ArrowField::new(
            "b",
            DataType::Struct(ArrowFields::from(vec![
                ArrowField::new("f1", DataType::Utf8, true),
                ArrowField::new("f2", DataType::Boolean, false),
                ArrowField::new("f3", DataType::Float32, false),
            ])),
            true,
        )]);
        assert_eq!(ArrowSchema::from(&projected), expected_arrow_schema);

        let projected = schema.project_by_ids(&[1, 2], false);
        let expected_arrow_schema = ArrowSchema::new(vec![ArrowField::new(
            "b",
            DataType::Struct(ArrowFields::from(vec![ArrowField::new(
                "f1",
                DataType::Utf8,
                true,
            )])),
            true,
        )]);
        assert_eq!(ArrowSchema::from(&projected), expected_arrow_schema);
    }

    #[test]
    fn test_schema_project_by_schema() {
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new(
                "b",
                DataType::Struct(ArrowFields::from(vec![
                    ArrowField::new("f1", DataType::Utf8, true),
                    ArrowField::new("f2", DataType::Boolean, false),
                    ArrowField::new("f3", DataType::Float32, false),
                ])),
                true,
            ),
            ArrowField::new("c", DataType::Float64, false),
            ArrowField::new("s", DataType::Utf8, false),
            ArrowField::new(
                "l",
                DataType::List(Arc::new(ArrowField::new("le", DataType::Int32, false))),
                false,
            ),
            ArrowField::new(
                "fixed_l",
                DataType::List(Arc::new(ArrowField::new("elem", DataType::Float32, false))),
                false,
            ),
            ArrowField::new(
                "d",
                DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                false,
            ),
        ]);
        let schema = Schema::try_from(&arrow_schema).unwrap();

        let projection = ArrowSchema::new(vec![
            ArrowField::new(
                "b",
                DataType::Struct(ArrowFields::from(vec![ArrowField::new(
                    "f1",
                    DataType::Utf8,
                    true,
                )])),
                true,
            ),
            ArrowField::new("s", DataType::Utf8, false),
            ArrowField::new(
                "l",
                DataType::List(Arc::new(ArrowField::new("le", DataType::Int32, false))),
                false,
            ),
            ArrowField::new(
                "fixed_l",
                DataType::List(Arc::new(ArrowField::new("elem", DataType::Float32, false))),
                false,
            ),
            ArrowField::new(
                "d",
                DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                false,
            ),
        ]);
        let projected = schema.project_by_schema(&projection).unwrap();

        assert_eq!(ArrowSchema::from(&projected), projection);
    }

    #[test]
    fn test_get_nested_field() {
        let arrow_schema = ArrowSchema::new(vec![ArrowField::new(
            "b",
            DataType::Struct(ArrowFields::from(vec![
                ArrowField::new("f1", DataType::Utf8, true),
                ArrowField::new("f2", DataType::Boolean, false),
                ArrowField::new("f3", DataType::Float32, false),
            ])),
            true,
        )]);
        let schema = Schema::try_from(&arrow_schema).unwrap();

        let field = schema.field("b.f2").unwrap();
        assert_eq!(field.data_type(), DataType::Boolean);
    }

    #[test]
    fn test_exclude_fields() {
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new(
                "b",
                DataType::Struct(ArrowFields::from(vec![
                    ArrowField::new("f1", DataType::Utf8, true),
                    ArrowField::new("f2", DataType::Boolean, false),
                    ArrowField::new("f3", DataType::Float32, false),
                ])),
                true,
            ),
            ArrowField::new("c", DataType::Float64, false),
        ]);
        let schema = Schema::try_from(&arrow_schema).unwrap();

        let projection = schema.project(&["a", "b.f2", "b.f3"]).unwrap();
        let excluded = schema.exclude(&projection).unwrap();

        let expected_arrow_schema = ArrowSchema::new(vec![
            ArrowField::new(
                "b",
                DataType::Struct(ArrowFields::from(vec![ArrowField::new(
                    "f1",
                    DataType::Utf8,
                    true,
                )])),
                true,
            ),
            ArrowField::new("c", DataType::Float64, false),
        ]);
        assert_eq!(ArrowSchema::from(&excluded), expected_arrow_schema);
    }

    #[test]
    fn test_intersection() {
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new(
                "b",
                DataType::Struct(ArrowFields::from(vec![
                    ArrowField::new("f1", DataType::Utf8, true),
                    ArrowField::new("f2", DataType::Boolean, false),
                    ArrowField::new("f3", DataType::Float32, false),
                ])),
                true,
            ),
            ArrowField::new("c", DataType::Float64, false),
        ]);
        let schema = Schema::try_from(&arrow_schema).unwrap();

        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new(
                "b",
                DataType::Struct(ArrowFields::from(vec![
                    ArrowField::new("f1", DataType::Utf8, true),
                    ArrowField::new("f2", DataType::Boolean, false),
                ])),
                true,
            ),
            ArrowField::new("c", DataType::Float64, false),
            ArrowField::new("d", DataType::Utf8, false),
        ]);
        let other = Schema::try_from(&arrow_schema).unwrap();

        let actual: ArrowSchema = (&schema.intersection(&other).unwrap()).into();

        let expected = ArrowSchema::new(vec![
            ArrowField::new(
                "b",
                DataType::Struct(ArrowFields::from(vec![
                    ArrowField::new("f1", DataType::Utf8, true),
                    ArrowField::new("f2", DataType::Boolean, false),
                ])),
                true,
            ),
            ArrowField::new("c", DataType::Float64, false),
        ]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_merge_schemas_and_assign_field_ids() {
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new(
                "b",
                DataType::Struct(ArrowFields::from(vec![
                    ArrowField::new("f1", DataType::Utf8, true),
                    ArrowField::new("f2", DataType::Boolean, false),
                    ArrowField::new("f3", DataType::Float32, false),
                ])),
                true,
            ),
            ArrowField::new("c", DataType::Float64, false),
        ]);
        let schema = Schema::try_from(&arrow_schema).unwrap();

        assert_eq!(schema.max_field_id(), Some(5));

        let to_merged_arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("d", DataType::Int32, false),
            ArrowField::new("e", DataType::Binary, false),
        ]);
        let to_merged = Schema::try_from(&to_merged_arrow_schema).unwrap();
        // It is already assigned with field ids.
        assert_eq!(to_merged.max_field_id(), Some(1));

        let mut merged = schema.merge(&to_merged).unwrap();
        assert_eq!(merged.max_field_id(), Some(5));

        let field = merged.field("d").unwrap();
        assert_eq!(field.id, -1);
        let field = merged.field("e").unwrap();
        assert_eq!(field.id, -1);

        // Need to explicitly assign field ids. Testing we can pass a larger
        // field id to set_field_id.
        merged.set_field_id(Some(7));
        let field = merged.field("d").unwrap();
        assert_eq!(field.id, 8);
        let field = merged.field("e").unwrap();
        assert_eq!(field.id, 9);
        assert_eq!(merged.max_field_id(), Some(9));
    }

    #[test]
    fn test_merge_arrow_schema() {
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new(
                "b",
                DataType::Struct(ArrowFields::from(vec![
                    ArrowField::new("f1", DataType::Utf8, true),
                    ArrowField::new("f2", DataType::Boolean, false),
                    ArrowField::new("f3", DataType::Float32, false),
                ])),
                true,
            ),
            ArrowField::new("c", DataType::Float64, false),
        ]);
        let schema = Schema::try_from(&arrow_schema).unwrap();

        assert_eq!(schema.max_field_id(), Some(5));

        let to_merged_arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("d", DataType::Int32, false),
            ArrowField::new("e", DataType::Binary, false),
        ]);
        let mut merged = schema.merge(&to_merged_arrow_schema).unwrap();
        merged.set_field_id(None);
        assert_eq!(merged.max_field_id(), Some(7));

        let field = merged.field("d").unwrap();
        assert_eq!(field.id, 6);
        let field = merged.field("e").unwrap();
        assert_eq!(field.id, 7);
    }

    #[test]
    fn test_merge_nested_field() {
        let arrow_schema1 = ArrowSchema::new(vec![ArrowField::new(
            "b",
            DataType::Struct(ArrowFields::from(vec![
                ArrowField::new(
                    "f1",
                    DataType::Struct(ArrowFields::from(vec![ArrowField::new(
                        "f11",
                        DataType::Utf8,
                        true,
                    )])),
                    true,
                ),
                ArrowField::new("f2", DataType::Float32, false),
            ])),
            true,
        )]);
        let schema1 = Schema::try_from(&arrow_schema1).unwrap();

        let arrow_schema2 = ArrowSchema::new(vec![ArrowField::new(
            "b",
            DataType::Struct(ArrowFields::from(vec![
                ArrowField::new(
                    "f1",
                    DataType::Struct(ArrowFields::from(vec![ArrowField::new(
                        "f22",
                        DataType::Utf8,
                        true,
                    )])),
                    true,
                ),
                ArrowField::new("f3", DataType::Float32, false),
            ])),
            true,
        )]);
        let schema2 = Schema::try_from(&arrow_schema2).unwrap();

        let expected_arrow_schema = ArrowSchema::new(vec![ArrowField::new(
            "b",
            DataType::Struct(ArrowFields::from(vec![
                ArrowField::new(
                    "f1",
                    DataType::Struct(ArrowFields::from(vec![
                        ArrowField::new("f11", DataType::Utf8, true),
                        ArrowField::new("f22", DataType::Utf8, true),
                    ])),
                    true,
                ),
                ArrowField::new("f2", DataType::Float32, false),
                ArrowField::new("f3", DataType::Float32, false),
            ])),
            true,
        )]);
        let mut expected_schema = Schema::try_from(&expected_arrow_schema).unwrap();
        expected_schema.fields[0]
            .child_mut("f1")
            .unwrap()
            .child_mut("f22")
            .unwrap()
            .id = 4;
        expected_schema.fields[0].child_mut("f2").unwrap().id = 3;

        let mut result = schema1.merge(&schema2).unwrap();
        result.set_field_id(None);
        assert_eq!(result, expected_schema);
    }

    #[test]
    fn test_field_by_id() {
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new(
                "b",
                DataType::Struct(ArrowFields::from(vec![
                    ArrowField::new("f1", DataType::Utf8, true),
                    ArrowField::new("f2", DataType::Boolean, false),
                    ArrowField::new("f3", DataType::Float32, false),
                ])),
                true,
            ),
            ArrowField::new("c", DataType::Float64, false),
        ]);
        let schema = Schema::try_from(&arrow_schema).unwrap();

        let field = schema.field_by_id(1).unwrap();
        assert_eq!(field.name, "b");

        let field = schema.field_by_id(3).unwrap();
        assert_eq!(field.name, "f2");
    }

    #[test]
    fn test_explain_difference() {
        let expected = ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new(
                "b",
                DataType::Struct(ArrowFields::from(vec![
                    ArrowField::new("f1", DataType::Utf8, true),
                    ArrowField::new("f2", DataType::Boolean, false),
                    ArrowField::new("f3", DataType::Float32, false),
                ])),
                true,
            ),
            ArrowField::new("c", DataType::Float64, false),
        ]);
        let expected = Schema::try_from(&expected).unwrap();

        let mismatched = ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new(
                "b",
                DataType::Struct(ArrowFields::from(vec![
                    ArrowField::new("f1", DataType::Utf8, true),
                    ArrowField::new("f3", DataType::Float32, false),
                ])),
                true,
            ),
            ArrowField::new("c", DataType::Float64, true),
        ]);
        let mismatched = Schema::try_from(&mismatched).unwrap();

        assert_eq!(
            mismatched.explain_difference(&expected, &SchemaCompareOptions::default()),
            Some(
                "`b` had mismatched children: fields did not match, missing=[b.f2], \
                  unexpected=[], `c` should have nullable=false but nullable=true"
                    .to_string()
            )
        );
    }

    #[test]
    fn test_schema_difference_subschema() {
        let expected = ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new(
                "b",
                DataType::Struct(ArrowFields::from(vec![
                    ArrowField::new("f1", DataType::Utf8, true),
                    ArrowField::new("f2", DataType::Boolean, false),
                    ArrowField::new("f3", DataType::Float32, false),
                ])),
                true,
            ),
            ArrowField::new("c", DataType::Float64, true),
        ]);
        let expected = Schema::try_from(&expected).unwrap();

        // Can omit nullable fields and subfields
        let subschema = ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new(
                "b",
                DataType::Struct(ArrowFields::from(vec![
                    ArrowField::new("f2", DataType::Boolean, false),
                    ArrowField::new("f3", DataType::Float32, false),
                ])),
                true,
            ),
        ]);
        let subschema = Schema::try_from(&subschema).unwrap();

        assert!(!subschema.compare_with_options(&expected, &SchemaCompareOptions::default()));
        assert_eq!(
            subschema.explain_difference(&expected, &SchemaCompareOptions::default()),
            Some(
                "fields did not match, missing=[c], unexpected=[], `b` had mismatched \
                 children: fields did not match, missing=[b.f1], unexpected=[]"
                    .to_string()
            )
        );
        let options = SchemaCompareOptions {
            allow_missing_if_nullable: true,
            ..Default::default()
        };
        assert!(subschema.compare_with_options(&expected, &options));
        let res = subschema.explain_difference(&expected, &options);
        assert!(res.is_none(), "Expected None, got {:?}", res);

        // Omitting non-nullable fields should fail
        let subschema = ArrowSchema::new(vec![ArrowField::new(
            "b",
            DataType::Struct(ArrowFields::from(vec![ArrowField::new(
                "f2",
                DataType::Boolean,
                false,
            )])),
            true,
        )]);
        let subschema = Schema::try_from(&subschema).unwrap();
        assert!(!subschema.compare_with_options(&expected, &options));
        assert_eq!(
            subschema.explain_difference(&expected, &options),
            Some(
                "fields did not match, missing=[a], unexpected=[], `b` had mismatched \
                 children: fields did not match, missing=[b.f3], unexpected=[]"
                    .to_string()
            )
        );

        let out_of_order = ArrowSchema::new(vec![
            ArrowField::new("c", DataType::Float64, true),
            ArrowField::new(
                "b",
                DataType::Struct(ArrowFields::from(vec![
                    ArrowField::new("f3", DataType::Float32, false),
                    ArrowField::new("f2", DataType::Boolean, false),
                    ArrowField::new("f1", DataType::Utf8, true),
                ])),
                true,
            ),
            ArrowField::new("a", DataType::Int32, false),
        ]);
        let out_of_order = Schema::try_from(&out_of_order).unwrap();
        assert!(!out_of_order.compare_with_options(&expected, &options));
        assert_eq!(
            subschema.explain_difference(&expected, &options),
            Some(
                "fields did not match, missing=[a], unexpected=[], `b` had mismatched \
                 children: fields did not match, missing=[b.f3], unexpected=[]"
                    .to_string()
            )
        );

        let options = SchemaCompareOptions {
            ignore_field_order: true,
            ..Default::default()
        };
        assert!(out_of_order.compare_with_options(&expected, &options));
        let res = out_of_order.explain_difference(&expected, &options);
        assert!(res.is_none(), "Expected None, got {:?}", res);
    }
}
