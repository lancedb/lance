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

//! Schema

use std::{
    collections::HashMap,
    fmt::{self, Debug, Formatter},
};

use arrow_array::RecordBatch;
use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};

use super::field::Field;
use crate::arrow::*;
use crate::{format::pb, io::object_reader::ObjectReader, Error, Result};

/// Lance Schema.
#[derive(Default, Debug, Clone)]
pub struct Schema {
    /// Top-level fields in the dataset.
    pub fields: Vec<Field>,
    /// Metadata of the schema
    pub metadata: HashMap<String, String>,
}

impl Schema {
    /// Project the columns over the schema.
    ///
    /// ```ignore
    /// let schema = Schema::from(...);
    /// let projected = schema.project(&["col1", "col2.sub_col3.field4"])?;
    /// ```
    pub fn project<T: AsRef<str>>(&self, columns: &[T]) -> Result<Self> {
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
            } else {
                return Err(Error::Schema {
                    message: format!("Column {} does not exist", col.as_ref()),
                });
            }
        }

        Ok(Self {
            fields: candidates,
            metadata: self.metadata.clone(),
        })
    }

    /// Check that the top level fields don't contain `.` in their names
    /// to distinguish from nested fields.
    pub(crate) fn validate(&self) -> Result<bool> {
        for field in self.fields.iter() {
            if field.name.contains('.') {
                return Err(Error::Schema{message:format!(
                    "Top level field {} cannot contain `.`. Maybe you meant to create a struct field?",
                    field.name.clone()
                )});
            }
        }
        Ok(true)
    }

    /// Intersection between two [`Schema`].
    pub fn intersection(&self, other: &Self) -> Result<Self> {
        let mut candidates: Vec<Field> = vec![];
        for field in other.fields.iter() {
            if let Some(candidate_field) = self.field(&field.name) {
                candidates.push(candidate_field.intersection(field)?);
            }
        }

        Ok(Self {
            fields: candidates,
            metadata: self.metadata.clone(),
        })
    }

    pub fn project_by_ids(&self, column_ids: &[i32]) -> Result<Self> {
        let protos: Vec<pb::Field> = self.into();

        let filtered_protos: Vec<pb::Field> = protos
            .iter()
            .filter(|p| column_ids.contains(&p.id))
            .cloned()
            .collect();
        Ok(Self::from(&filtered_protos))
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
            metadata: HashMap::default(),
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

    pub(crate) fn field_id(&self, column: &str) -> Result<i32> {
        self.field(column)
            .map(|f| f.id)
            .ok_or_else(|| Error::Schema {
                message: "Vector column not in schema".to_string(),
            })
    }

    /// Recursively collect all the field IDs,
    pub(crate) fn field_ids(&self) -> Vec<i32> {
        // TODO: make a tree traversal iterator.
        let protos: Vec<pb::Field> = self.into();
        protos.iter().map(|f| f.id).collect()
    }

    pub(crate) fn mut_field_by_id(&mut self, id: i32) -> Option<&mut Field> {
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

    pub(crate) fn max_field_id(&self) -> Option<i32> {
        self.fields.iter().map(|f| f.max_id()).max()
    }

    /// Load dictionary value array from manifest files.
    pub(crate) async fn load_dictionary<'a>(&mut self, reader: &dyn ObjectReader) -> Result<()> {
        for field in self.fields.as_mut_slice() {
            field.load_dictionary(reader).await?;
        }
        Ok(())
    }

    /// Recursively attach set up dictionary values to the dictionary fields.
    pub(crate) fn set_dictionary(&mut self, batch: &RecordBatch) -> Result<()> {
        for field in self.fields.as_mut_slice() {
            let column = batch
                .column_by_name(&field.name)
                .ok_or_else(|| Error::Schema {
                    message: format!("column '{}' does not exist in the record batch", field.name),
                })?;
            field.set_dictionary(column);
        }
        Ok(())
    }

    fn set_field_id(&mut self) {
        let mut current_id = self.max_field_id().unwrap_or(-1) + 1;
        self.fields
            .iter_mut()
            .for_each(|f| f.set_id(-1, &mut current_id));
    }

    fn reset_id(&mut self) {
        self.fields.iter_mut().for_each(|f| f.reset_id());
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
        let mut schema = Self {
            fields: merged_fields,
            metadata,
        };
        schema.set_field_id();
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
        schema.set_field_id();

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

/// Convert list of protobuf `Field` to a Schema.
impl From<&Vec<pb::Field>> for Schema {
    fn from(fields: &Vec<pb::Field>) -> Self {
        let mut schema = Self {
            fields: vec![],
            metadata: HashMap::default(),
        };

        fields.iter().for_each(|f| {
            if f.parent_id == -1 {
                schema.fields.push(Field::from(f));
            } else {
                let parent = schema.mut_field_by_id(f.parent_id).unwrap();
                parent.children.push(Field::from(f));
            }
        });

        schema
    }
}

/// Convert list of protobuf `Field` and Metadata to a Schema.
impl From<(&Vec<pb::Field>, HashMap<String, Vec<u8>>)> for Schema {
    fn from((fields, metadata): (&Vec<pb::Field>, HashMap<String, Vec<u8>>)) -> Self {
        let lance_metadata = metadata
            .into_iter()
            .map(|(key, value)| {
                let string_value = String::from_utf8_lossy(&value).to_string();
                (key, string_value)
            })
            .collect();

        let schema_with_fields = Self::from(fields);
        Self {
            fields: schema_with_fields.fields,
            metadata: lance_metadata,
        }
    }
}

/// Convert a Schema to a list of protobuf Field.
impl From<&Schema> for Vec<pb::Field> {
    fn from(schema: &Schema) -> Self {
        let mut protos: Self = vec![];
        schema.fields.iter().for_each(|f| {
            protos.extend(Self::from(f));
        });
        protos
    }
}

/// Convert a Schema to a list of protobuf Field and Metadata
impl From<&Schema> for (Vec<pb::Field>, HashMap<String, Vec<u8>>) {
    fn from(schema: &Schema) -> Self {
        let fields: Vec<pb::Field> = schema.into();
        let pb_metadata = schema
            .metadata
            .clone()
            .into_iter()
            .map(|(key, value)| (key, value.into_bytes()))
            .collect();
        (fields, pb_metadata)
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
        let schema = Schema::try_from(&arrow_schema).unwrap();
        let projected = schema.project_by_ids(&[1, 2, 4, 5]).unwrap();

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
    fn test_schema_set_ids() {
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

        let protos: Vec<pb::Field> = (&schema).into();
        assert_eq!(
            protos.iter().map(|p| p.id).collect::<Vec<_>>(),
            (0..6).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_schema_metadata() {
        let mut metadata: HashMap<String, String> = HashMap::new();
        metadata.insert(String::from("k1"), String::from("v1"));
        metadata.insert(String::from("k2"), String::from("v2"));

        let arrow_schema = ArrowSchema::new_with_metadata(
            vec![ArrowField::new("a", DataType::Int32, false)],
            metadata,
        );

        let expected_schema = Schema::try_from(&arrow_schema).unwrap();
        let (fields, meta): (Vec<pb::Field>, HashMap<String, Vec<u8>>) = (&expected_schema).into();

        let schema = Schema::from((&fields, meta));
        assert_eq!(expected_schema, schema);
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

        let merged = schema.merge(&to_merged).unwrap();
        assert_eq!(merged.max_field_id(), Some(7));

        let field = merged.field("d").unwrap();
        assert_eq!(field.id, 6);
        let field = merged.field("e").unwrap();
        assert_eq!(field.id, 7);
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
        let merged = schema.merge(&to_merged_arrow_schema).unwrap();
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

        let result = schema1.merge(&schema2).unwrap();
        assert_eq!(result, expected_schema);
    }
}
