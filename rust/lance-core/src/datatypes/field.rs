// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Schema Field

use std::{
    cmp::max,
    collections::{HashMap, HashSet},
    fmt,
    sync::Arc,
};

use arrow_array::{
    cast::AsArray,
    types::{
        Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
    },
    ArrayRef,
};
use arrow_schema::{DataType, Field as ArrowField};
use deepsize::DeepSizeOf;
use lance_arrow::{bfloat16::ARROW_EXT_NAME_KEY, *};
use snafu::{location, Location};

use super::{Dictionary, LogicalType};
use crate::{Error, Result};

#[derive(Default)]
pub struct SchemaCompareOptions {
    /// Should the metadata be compared (default false)
    pub compare_metadata: bool,
    /// Should the dictionaries be compared (default false)
    pub compare_dictionary: bool,
    /// Should the field ids be compared (default false)
    pub compare_field_ids: bool,
}
/// Encoding enum.
#[derive(Debug, Clone, PartialEq, Eq, DeepSizeOf)]
pub enum Encoding {
    /// Plain encoding.
    Plain,
    /// Binary encoding.
    VarBinary,
    /// Dictionary encoding.
    Dictionary,
    /// RLE encoding.
    RLE,
}

/// Lance Schema Field
///
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct Field {
    pub name: String,
    pub id: i32,
    // TODO: Find way to move these next three fields to private
    pub parent_id: i32,
    pub logical_type: LogicalType,
    pub metadata: HashMap<String, String>,
    pub encoding: Option<Encoding>,
    pub nullable: bool,

    pub children: Vec<Field>,

    /// Dictionary value array if this field is dictionary.
    pub dictionary: Option<Dictionary>,
}

impl Field {
    /// Returns arrow data type.
    pub fn data_type(&self) -> DataType {
        match &self.logical_type {
            lt if lt.is_list() => DataType::List(Arc::new(ArrowField::from(&self.children[0]))),
            lt if lt.is_large_list() => {
                DataType::LargeList(Arc::new(ArrowField::from(&self.children[0])))
            }
            lt if lt.is_struct() => {
                DataType::Struct(self.children.iter().map(ArrowField::from).collect())
            }
            lt => DataType::try_from(lt).unwrap(),
        }
    }

    pub fn has_dictionary_types(&self) -> bool {
        matches!(self.data_type(), DataType::Dictionary(_, _))
            || self.children.iter().any(Self::has_dictionary_types)
    }

    fn explain_differences(
        &self,
        expected: &Self,
        options: &SchemaCompareOptions,
        path: Option<&str>,
    ) -> Vec<String> {
        let mut differences = Vec::new();
        let self_name = path
            .map(|path| {
                let mut self_name = path.to_owned();
                self_name.push('.');
                self_name.push_str(&self.name);
                self_name
            })
            .unwrap_or_else(|| self.name.clone());
        if self.name != expected.name {
            let expected_path = path
                .map(|path| {
                    let mut expected_path = path.to_owned();
                    expected_path.push('.');
                    expected_path.push_str(&expected.name);
                    expected_path
                })
                .unwrap_or_else(|| expected.name.clone());
            differences.push(format!(
                "expected name '{}' but name was '{}'",
                expected_path, self_name
            ));
        }
        if options.compare_field_ids && self.id != expected.id {
            differences.push(format!(
                "`{}` should have id {} but id was {}",
                self_name, expected.id, self.id
            ));
        }
        if self.logical_type != expected.logical_type {
            differences.push(format!(
                "`{}` should have type {} but type was {}",
                self_name, expected.logical_type, self.logical_type
            ));
        }
        if self.nullable != expected.nullable {
            differences.push(format!(
                "`{}` should have nullable={} but nullable={}",
                self_name, expected.nullable, self.nullable
            ))
        }
        if options.compare_dictionary && self.dictionary != expected.dictionary {
            differences.push(format!(
                "dictionary for `{}` did not match expected dictionary",
                self_name
            ));
        }
        if options.compare_metadata && self.metadata != expected.metadata {
            differences.push(format!(
                "metadata for `{}` did not match expected metadata",
                self_name
            ));
        }
        if self.children.len() != expected.children.len()
            || !self
                .children
                .iter()
                .zip(expected.children.iter())
                .all(|(child, expected)| child.name == expected.name)
        {
            let self_children = self
                .children
                .iter()
                .map(|child| child.name.clone())
                .collect::<HashSet<_>>();
            let expected_children = expected
                .children
                .iter()
                .map(|child| child.name.clone())
                .collect::<HashSet<_>>();
            let missing = expected_children
                .difference(&self_children)
                .cloned()
                .collect::<Vec<_>>();
            let unexpected = self_children
                .difference(&expected_children)
                .cloned()
                .collect::<Vec<_>>();
            if missing.is_empty() && unexpected.is_empty() {
                differences.push(format!(
                    "`{}` field order mismatch, expected [{}] but was [{}]",
                    self_name,
                    expected
                        .children
                        .iter()
                        .map(|child| child.name.clone())
                        .collect::<Vec<_>>()
                        .join(", "),
                    self.children
                        .iter()
                        .map(|child| child.name.clone())
                        .collect::<Vec<_>>()
                        .join(", "),
                ));
            } else {
                differences.push(format!(
                    "`{}` had mismatched children, missing=[{}] unexpected=[{}]",
                    self_name,
                    missing.join(", "),
                    unexpected.join(", ")
                ));
            }
        } else {
            differences.extend(self.children.iter().zip(expected.children.iter()).flat_map(
                |(child, expected_child)| {
                    child.explain_differences(expected_child, options, Some(&self_name))
                },
            ));
        }
        differences
    }

    pub fn explain_difference(
        &self,
        expected: &Self,
        options: &SchemaCompareOptions,
    ) -> Option<String> {
        let differences = self.explain_differences(expected, options, None);
        if differences.is_empty() {
            None
        } else {
            Some(differences.join(", "))
        }
    }

    pub fn compare_with_options(&self, expected: &Self, options: &SchemaCompareOptions) -> bool {
        if self.children.len() != expected.children.len() {
            false
        } else {
            self.name == expected.name
                && self.logical_type == expected.logical_type
                && self.nullable == expected.nullable
                && self.children.len() == expected.children.len()
                && self
                    .children
                    .iter()
                    .zip(&expected.children)
                    .all(|(left, right)| left.compare_with_options(right, options))
                && (!options.compare_field_ids || self.id == expected.id)
                && (!options.compare_dictionary || self.dictionary == expected.dictionary)
                && (!options.compare_metadata || self.metadata == expected.metadata)
        }
    }

    pub fn extension_name(&self) -> Option<&str> {
        self.metadata.get(ARROW_EXT_NAME_KEY).map(String::as_str)
    }

    pub fn child(&self, name: &str) -> Option<&Self> {
        self.children.iter().find(|f| f.name == name)
    }

    pub fn child_mut(&mut self, name: &str) -> Option<&mut Self> {
        self.children.iter_mut().find(|f| f.name == name)
    }

    /// Attach the Dictionary's value array, so that we can later serialize
    /// the dictionary to the manifest.
    pub fn set_dictionary_values(&mut self, arr: &ArrayRef) {
        assert!(self.data_type().is_dictionary());
        // offset / length are set to 0 and recomputed when the dictionary is persisted to disk
        self.dictionary = Some(Dictionary {
            offset: 0,
            length: 0,
            values: Some(arr.clone()),
        });
    }

    pub fn set_dictionary(&mut self, arr: &ArrayRef) {
        let data_type = self.data_type();
        match data_type {
            DataType::Dictionary(key_type, _) => match key_type.as_ref() {
                DataType::Int8 => {
                    self.set_dictionary_values(arr.as_dictionary::<Int8Type>().values())
                }
                DataType::Int16 => {
                    self.set_dictionary_values(arr.as_dictionary::<Int16Type>().values())
                }
                DataType::Int32 => {
                    self.set_dictionary_values(arr.as_dictionary::<Int32Type>().values())
                }
                DataType::Int64 => {
                    self.set_dictionary_values(arr.as_dictionary::<Int64Type>().values())
                }
                DataType::UInt8 => {
                    self.set_dictionary_values(arr.as_dictionary::<UInt8Type>().values())
                }
                DataType::UInt16 => {
                    self.set_dictionary_values(arr.as_dictionary::<UInt16Type>().values())
                }
                DataType::UInt32 => {
                    self.set_dictionary_values(arr.as_dictionary::<UInt32Type>().values())
                }
                DataType::UInt64 => {
                    self.set_dictionary_values(arr.as_dictionary::<UInt64Type>().values())
                }
                _ => {
                    panic!("Unsupported dictionary key type: {}", key_type);
                }
            },
            DataType::Struct(subfields) => {
                for (i, f) in subfields.iter().enumerate() {
                    let lance_field = self
                        .children
                        .iter_mut()
                        .find(|c| c.name == *f.name())
                        .unwrap();
                    let struct_arr = arr.as_struct();
                    lance_field.set_dictionary(struct_arr.column(i));
                }
            }
            DataType::List(_) => {
                let list_arr = arr.as_list::<i32>();
                self.children[0].set_dictionary(list_arr.values());
            }
            DataType::LargeList(_) => {
                let list_arr = arr.as_list::<i64>();
                self.children[0].set_dictionary(list_arr.values());
            }
            _ => {
                // Field types that don't support dictionaries
            }
        }
    }

    pub fn sub_field(&self, path_components: &[&str]) -> Option<&Self> {
        if path_components.is_empty() {
            Some(self)
        } else {
            let first = path_components[0];
            self.children
                .iter()
                .find(|c| c.name == first)
                .and_then(|c| c.sub_field(&path_components[1..]))
        }
    }

    pub fn project(&self, path_components: &[&str]) -> Result<Self> {
        let mut f = Self {
            name: self.name.clone(),
            id: self.id,
            parent_id: self.parent_id,
            logical_type: self.logical_type.clone(),
            metadata: self.metadata.clone(),
            encoding: self.encoding.clone(),
            nullable: self.nullable,
            children: vec![],
            dictionary: self.dictionary.clone(),
        };
        if path_components.is_empty() {
            // Project stops here, copy all the remaining children.
            f.children.clone_from(&self.children)
        } else {
            let first = path_components[0];
            for c in self.children.as_slice() {
                if c.name == first {
                    let projected = c.project(&path_components[1..])?;
                    f.children.push(projected);
                    break;
                }
            }
        }
        Ok(f)
    }

    /// Create a new field by removing all fields that do not match the filter.
    ///
    /// If a child field matches the filter then the parent will be kept even if
    /// it does not match the filter.
    ///
    /// Returns None if the field itself does not match the filter.
    pub fn project_by_filter<F: Fn(&Self) -> bool>(&self, filter: &F) -> Option<Self> {
        let children = self
            .children
            .iter()
            .filter_map(|c| c.project_by_filter(filter))
            .collect::<Vec<_>>();
        if !children.is_empty() || filter(self) {
            Some(Self {
                children,
                ..self.clone()
            })
        } else {
            None
        }
    }

    /// Create a new field by selecting fields by their ids.
    ///
    /// If a field has it's id in the list of ids then it will be included
    /// in the new field. If a field is selected, all of it's parents will be
    /// and all of it's children will be included.
    ///
    /// For example, for the schema:
    ///
    /// ```text
    /// 0: x struct {
    ///     1: y int32
    ///     2: l list {
    ///         3: z int32
    ///     }
    /// }
    /// ```
    ///
    /// If the ids are `[2]`, then this will include the parent `0` and the
    /// child `3`.
    pub(crate) fn project_by_ids(&self, ids: &[i32]) -> Option<Self> {
        let children = self
            .children
            .iter()
            .filter_map(|c| c.project_by_ids(ids))
            .collect::<Vec<_>>();
        if ids.contains(&self.id) {
            Some(self.clone())
        } else if !children.is_empty() {
            Some(Self {
                children,
                ..self.clone()
            })
        } else {
            None
        }
    }

    /// Project by a field.
    ///
    pub fn project_by_field(&self, other: &Self) -> Result<Self> {
        if self.name != other.name {
            return Err(Error::Schema {
                message: format!(
                    "Attempt to project field by different names: {} and {}",
                    self.name, other.name,
                ),
                location: location!(),
            });
        };

        match (self.data_type(), other.data_type()) {
            (DataType::Boolean, DataType::Boolean) => Ok(self.clone()),
            (dt, other_dt)
                if (dt.is_primitive() && other_dt.is_primitive())
                    || (dt.is_binary_like() && other_dt.is_binary_like()) =>
            {
                if dt != other_dt {
                    return Err(Error::Schema {
                        message: format!(
                            "Attempt to project field by different types: {} and {}",
                            dt, other_dt,
                        ),
                        location: location!(),
                    });
                }
                Ok(self.clone())
            }
            (DataType::Struct(_), DataType::Struct(_)) => {
                let mut fields = vec![];
                for other_field in other.children.iter() {
                    let Some(child) = self.child(&other_field.name) else {
                        return Err(Error::Schema {
                            message: format!(
                                "Attempt to project non-existed field: {} on {}",
                                other_field.name, self,
                            ),
                            location: location!(),
                        });
                    };
                    fields.push(child.project_by_field(other_field)?);
                }
                let mut cloned = self.clone();
                cloned.children = fields;
                Ok(cloned)
            }
            (DataType::List(_), DataType::List(_))
            | (DataType::LargeList(_), DataType::LargeList(_)) => {
                let projected = self.children[0].project_by_field(&other.children[0])?;
                let mut cloned = self.clone();
                cloned.children = vec![projected];
                Ok(cloned)
            }
            (DataType::FixedSizeList(dt, n), DataType::FixedSizeList(other_dt, m))
                if dt == other_dt && n == m =>
            {
                Ok(self.clone())
            }
            (
                DataType::Dictionary(self_key, self_value),
                DataType::Dictionary(other_key, other_value),
            ) if self_key == other_key && self_value == other_value => Ok(self.clone()),
            (DataType::Null, DataType::Null) => Ok(self.clone()),
            (DataType::FixedSizeBinary(self_width), DataType::FixedSizeBinary(other_width))
                if self_width == other_width =>
            {
                Ok(self.clone())
            }
            _ => Err(Error::Schema {
                message: format!(
                    "Attempt to project incompatible fields: {} and {}",
                    self, other
                ),
                location: location!(),
            }),
        }
    }

    /// Intersection of two [`Field`]s.
    ///
    pub fn intersection(&self, other: &Self) -> Result<Self> {
        if self.name != other.name {
            return Err(Error::Arrow {
                message: format!(
                    "Attempt to intersect different fields: {} and {}",
                    self.name, other.name,
                ),
                location: location!(),
            });
        }
        let self_type = self.data_type();
        let other_type = other.data_type();
        if self_type.is_struct() && other_type.is_struct() {
            let children = self
                .children
                .iter()
                .filter_map(|c| {
                    if let Some(other_child) = other.child(&c.name) {
                        let intersection = c.intersection(other_child).ok()?;
                        Some(intersection)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            let f = Self {
                name: self.name.clone(),
                id: self.id,
                parent_id: self.parent_id,
                logical_type: self.logical_type.clone(),
                metadata: self.metadata.clone(),
                encoding: self.encoding.clone(),
                nullable: self.nullable,
                children,
                dictionary: self.dictionary.clone(),
            };
            return Ok(f);
        }

        if self_type != other_type || self.name != other.name {
            return Err(Error::Arrow {
                message: format!(
                    "Attempt to intersect different fields: ({}, {}) and ({}, {})",
                    self.name, self_type, other.name, other_type
                ),
                location: location!(),
            });
        }

        Ok(self.clone())
    }

    pub fn exclude(&self, other: &Self) -> Option<Self> {
        if !self.data_type().is_nested() {
            return None;
        }
        let children = self
            .children
            .iter()
            .map(|c| {
                if let Some(other_child) = other.child(&c.name) {
                    c.exclude(other_child)
                } else {
                    Some(c.clone())
                }
            })
            .filter(Option::is_some)
            .flatten()
            .collect::<Vec<_>>();
        if children.is_empty() {
            None
        } else {
            Some(Self {
                name: self.name.clone(),
                id: self.id,
                parent_id: self.parent_id,
                logical_type: self.logical_type.clone(),
                metadata: self.metadata.clone(),
                encoding: self.encoding.clone(),
                nullable: self.nullable,
                children,
                dictionary: self.dictionary.clone(),
            })
        }
    }

    /// Merge the children of other field into this one.
    pub(super) fn merge(&mut self, other: &Self) -> Result<()> {
        match (self.data_type(), other.data_type()) {
            (DataType::Struct(_), DataType::Struct(_)) => {
                for other_child in other.children.as_slice() {
                    if let Some(field) = self.child_mut(&other_child.name) {
                        field.merge(other_child)?;
                    } else {
                        self.children.push(other_child.clone());
                    }
                }
            }
            (DataType::List(_), DataType::List(_))
            | (DataType::LargeList(_), DataType::LargeList(_)) => {
                self.children[0].merge(&other.children[0])?;
            }
            (
                DataType::FixedSizeList(_, self_list_size),
                DataType::FixedSizeList(_, other_list_size),
            ) if self_list_size == other_list_size => {
                // do nothing
            }
            (DataType::FixedSizeBinary(self_size), DataType::FixedSizeBinary(other_size))
                if self_size == other_size =>
            {
                // do nothing
            }
            _ => {
                if self.data_type() != other.data_type() {
                    return Err(Error::Schema {
                        message: format!(
                            "Attempt to merge incompatible fields: {} and {}",
                            self, other
                        ),
                        location: location!(),
                    });
                }
            }
        }
        Ok(())
    }

    // Get the max field id of itself and all children.
    pub(super) fn max_id(&self) -> i32 {
        max(
            self.id,
            self.children.iter().map(|c| c.max_id()).max().unwrap_or(-1),
        )
    }

    /// Recursively set field ID and parent ID for this field and all its children.
    pub fn set_id(&mut self, parent_id: i32, id_seed: &mut i32) {
        self.parent_id = parent_id;
        if self.id < 0 {
            self.id = *id_seed;
            *id_seed += 1;
        }
        self.children
            .iter_mut()
            .for_each(|f| f.set_id(self.id, id_seed));
    }

    /// Recursively reset field ID for this field and all its children.
    pub(super) fn reset_id(&mut self) {
        self.id = -1;
        self.children.iter_mut().for_each(Self::reset_id);
    }

    pub fn field_by_id(&self, id: impl Into<i32>) -> Option<&Self> {
        let id = id.into();
        for child in self.children.as_slice() {
            if child.id == id {
                return Some(child);
            }
            if let Some(grandchild) = child.field_by_id(id) {
                return Some(grandchild);
            }
        }
        None
    }

    // Find any nested child with a specific field id
    pub(super) fn mut_field_by_id(&mut self, id: impl Into<i32>) -> Option<&mut Self> {
        let id = id.into();
        for child in self.children.as_mut_slice() {
            if child.id == id {
                return Some(child);
            }
            if let Some(grandchild) = child.mut_field_by_id(id) {
                return Some(grandchild);
            }
        }
        None
    }
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Field(id={}, name={}, type={}",
            self.id, self.name, self.logical_type.0,
        )?;

        if let Some(dictionary) = &self.dictionary {
            write!(f, ", dictionary={:?}", dictionary)?;
        }

        if !self.children.is_empty() {
            write!(f, ", children=[")?;
            for child in self.children.iter() {
                write!(f, "{}, ", child)?;
            }
            write!(f, "]")?;
        }

        write!(f, ")")
    }
}

impl TryFrom<&ArrowField> for Field {
    type Error = Error;

    fn try_from(field: &ArrowField) -> Result<Self> {
        let children = match field.data_type() {
            DataType::Struct(children) => children
                .iter()
                .map(|f| Self::try_from(f.as_ref()))
                .collect::<Result<_>>()?,
            DataType::List(item) => vec![Self::try_from(item.as_ref())?],
            DataType::LargeList(item) => vec![Self::try_from(item.as_ref())?],
            _ => vec![],
        };
        Ok(Self {
            id: -1,
            parent_id: -1,
            name: field.name().clone(),
            logical_type: LogicalType::try_from(field.data_type())?,
            encoding: match field.data_type() {
                dt if dt.is_fixed_stride() => Some(Encoding::Plain),
                dt if dt.is_binary_like() => Some(Encoding::VarBinary),
                DataType::Dictionary(_, _) => Some(Encoding::Dictionary),
                // Use plain encoder to store the offsets of list.
                DataType::List(_) | DataType::LargeList(_) => Some(Encoding::Plain),
                _ => None,
            },
            metadata: field.metadata().clone(),
            nullable: field.is_nullable(),
            children,
            dictionary: None,
        })
    }
}

impl TryFrom<ArrowField> for Field {
    type Error = Error;

    fn try_from(field: ArrowField) -> Result<Self> {
        Self::try_from(&field)
    }
}

impl From<&Field> for ArrowField {
    fn from(field: &Field) -> Self {
        let out = Self::new(&field.name, field.data_type(), field.nullable);
        out.with_metadata(field.metadata.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{DictionaryArray, StringArray, UInt32Array};
    use arrow_schema::{Fields, TimeUnit};

    #[test]
    fn arrow_field_to_field() {
        for (name, data_type) in [
            ("null", DataType::Null),
            ("bool", DataType::Boolean),
            ("int8", DataType::Int8),
            ("uint8", DataType::UInt8),
            ("int16", DataType::Int16),
            ("uint16", DataType::UInt16),
            ("int32", DataType::Int32),
            ("uint32", DataType::UInt32),
            ("int64", DataType::Int64),
            ("uint64", DataType::UInt64),
            ("float16", DataType::Float16),
            ("float32", DataType::Float32),
            ("float64", DataType::Float64),
            ("decimal128:7:3", DataType::Decimal128(7, 3)),
            ("timestamp:s:-", DataType::Timestamp(TimeUnit::Second, None)),
            (
                "timestamp:ms:-",
                DataType::Timestamp(TimeUnit::Millisecond, None),
            ),
            (
                "timestamp:us:-",
                DataType::Timestamp(TimeUnit::Microsecond, None),
            ),
            (
                "timestamp:ns:-",
                DataType::Timestamp(TimeUnit::Nanosecond, None),
            ),
            (
                "timestamp:s:America/New_York",
                DataType::Timestamp(TimeUnit::Second, Some("America/New_York".into())),
            ),
            ("time32:s", DataType::Time32(TimeUnit::Second)),
            ("time32:ms", DataType::Time32(TimeUnit::Millisecond)),
            ("time64:us", DataType::Time64(TimeUnit::Microsecond)),
            ("time64:ns", DataType::Time64(TimeUnit::Nanosecond)),
            ("duration:s", DataType::Duration(TimeUnit::Second)),
            ("duration:ms", DataType::Duration(TimeUnit::Millisecond)),
            ("duration:us", DataType::Duration(TimeUnit::Microsecond)),
            ("duration:ns", DataType::Duration(TimeUnit::Nanosecond)),
            ("fixed_size_binary:100", DataType::FixedSizeBinary(100)),
            (
                "fixed_size_list:int32:10",
                DataType::FixedSizeList(
                    Arc::new(ArrowField::new("item", DataType::Int32, true)),
                    10,
                ),
            ),
        ] {
            let arrow_field = ArrowField::new(name, data_type.clone(), true);
            let field = Field::try_from(&arrow_field).unwrap();
            assert_eq!(field.name, name);
            assert_eq!(field.data_type(), data_type);
            assert_eq!(ArrowField::from(&field), arrow_field);
        }
    }

    #[test]
    fn test_nested_types() {
        assert_eq!(
            LogicalType::try_from(&DataType::List(Arc::new(ArrowField::new(
                "item",
                DataType::Binary,
                false
            ))))
            .unwrap()
            .0,
            "list"
        );
        assert_eq!(
            LogicalType::try_from(&DataType::List(Arc::new(ArrowField::new(
                "item",
                DataType::Struct(Fields::empty()),
                false
            ))))
            .unwrap()
            .0,
            "list.struct"
        );
        assert_eq!(
            LogicalType::try_from(&DataType::Struct(Fields::from(vec![ArrowField::new(
                "item",
                DataType::Binary,
                false
            )])))
            .unwrap()
            .0,
            "struct"
        );
    }

    #[test]
    fn struct_field() {
        let arrow_field = ArrowField::new(
            "struct",
            DataType::Struct(Fields::from(vec![ArrowField::new(
                "a",
                DataType::Int32,
                true,
            )])),
            false,
        );
        let field = Field::try_from(&arrow_field).unwrap();
        assert_eq!(field.name, "struct");
        assert_eq!(&field.data_type(), arrow_field.data_type());
        assert_eq!(ArrowField::from(&field), arrow_field);
    }

    #[test]
    fn test_project_by_field_null_type() {
        let f1: Field = ArrowField::new("a", DataType::Null, true)
            .try_into()
            .unwrap();
        let f2: Field = ArrowField::new("a", DataType::Null, true)
            .try_into()
            .unwrap();
        let p1 = f1.project_by_field(&f2).unwrap();

        assert_eq!(p1, f1);

        let f3: Field = ArrowField::new("b", DataType::Null, true)
            .try_into()
            .unwrap();
        assert!(f1.project_by_field(&f3).is_err());

        let f4: Field = ArrowField::new("a", DataType::Int32, true)
            .try_into()
            .unwrap();
        assert!(f1.project_by_field(&f4).is_err());
    }

    #[test]
    fn test_field_intersection() {
        let f1: Field = ArrowField::new("a", DataType::Int32, true)
            .try_into()
            .unwrap();
        let f2: Field = ArrowField::new("a", DataType::Int32, true)
            .try_into()
            .unwrap();
        let i1 = f1.intersection(&f2).unwrap();

        assert_eq!(i1, f1);

        let f3: Field = ArrowField::new("b", DataType::Int32, true)
            .try_into()
            .unwrap();
        assert!(f1.intersection(&f3).is_err());
    }

    #[test]
    fn test_struct_field_intersection() {
        let f1: Field = ArrowField::new(
            "a",
            DataType::Struct(Fields::from(vec![
                ArrowField::new("b", DataType::Int32, true),
                ArrowField::new("c", DataType::Int32, true),
            ])),
            true,
        )
        .try_into()
        .unwrap();
        let f2: Field = ArrowField::new(
            "a",
            DataType::Struct(Fields::from(vec![
                ArrowField::new("c", DataType::Int32, true),
                ArrowField::new("a", DataType::Int32, true),
            ])),
            true,
        )
        .try_into()
        .unwrap();
        let actual = f1.intersection(&f2).unwrap();

        let expected: Field = ArrowField::new(
            "a",
            DataType::Struct(Fields::from(vec![ArrowField::new(
                "c",
                DataType::Int32,
                true,
            )])),
            true,
        )
        .try_into()
        .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_compare() {
        let opts = SchemaCompareOptions::default();

        let mut expected: Field = ArrowField::new(
            "a",
            DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
            true,
        )
        .try_into()
        .unwrap();
        let keys = UInt32Array::from_iter_values(vec![0, 1]);
        let values: ArrayRef = Arc::new(StringArray::from_iter_values([
            "a".to_string(),
            "b".to_string(),
        ]));
        let dictionary: ArrayRef = Arc::new(DictionaryArray::new(keys, values));
        expected.set_dictionary(&dictionary);

        let no_dictionary: Field = ArrowField::new(
            "a",
            DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
            true,
        )
        .try_into()
        .unwrap();

        // By default, do not compare dictionary
        assert!(no_dictionary.compare_with_options(&expected, &opts));

        let compare_dict = SchemaCompareOptions {
            compare_dictionary: true,
            ..Default::default()
        };
        assert!(!no_dictionary.compare_with_options(&expected, &compare_dict));

        let metadata = HashMap::<_, _>::from_iter(vec![("foo".to_string(), "bar".to_string())]);
        let expected: Field = ArrowField::new("a", DataType::UInt32, true)
            .with_metadata(metadata)
            .try_into()
            .unwrap();

        let no_metadata: Field = ArrowField::new("a", DataType::UInt32, true)
            .try_into()
            .unwrap();

        // By default, do not compare metadata
        assert!(no_metadata.compare_with_options(&expected, &opts));

        let compare_metadata = SchemaCompareOptions {
            compare_metadata: true,
            ..Default::default()
        };
        assert!(!no_metadata.compare_with_options(&expected, &compare_metadata));

        let mut expected: Field = ArrowField::new("a", DataType::UInt32, true)
            .try_into()
            .unwrap();
        let mut seed = 0;
        expected.set_id(-1, &mut seed);

        let no_id: Field = ArrowField::new("a", DataType::UInt32, true)
            .try_into()
            .unwrap();
        // Do not compare ids by default
        assert!(no_id.compare_with_options(&expected, &opts));

        let compare_ids = SchemaCompareOptions {
            compare_field_ids: true,
            ..Default::default()
        };
        assert!(!no_id.compare_with_options(&expected, &compare_ids));
    }

    #[test]
    fn test_explain_difference() {
        let expected: Field = ArrowField::new(
            "a",
            DataType::Struct(Fields::from(vec![
                ArrowField::new("b", DataType::Int32, true),
                ArrowField::new("c", DataType::Int32, true),
            ])),
            true,
        )
        .try_into()
        .unwrap();

        let opts = SchemaCompareOptions::default();
        assert_eq!(expected.explain_difference(&expected, &opts), None);

        let wrong_name: Field = ArrowField::new(
            "b",
            DataType::Struct(Fields::from(vec![
                ArrowField::new("b", DataType::Int32, true),
                ArrowField::new("c", DataType::Int32, true),
            ])),
            true,
        )
        .try_into()
        .unwrap();

        assert_eq!(
            wrong_name.explain_difference(&expected, &opts),
            Some("expected name 'a' but name was 'b'".to_string())
        );

        let wrong_child: Field = ArrowField::new(
            "a",
            DataType::Struct(Fields::from(vec![
                ArrowField::new("b", DataType::Int32, false),
                ArrowField::new("c", DataType::Int32, true),
            ])),
            true,
        )
        .try_into()
        .unwrap();
        assert_eq!(
            wrong_child.explain_difference(&expected, &opts),
            Some("`a.b` should have nullable=true but nullable=false".to_string())
        );

        let mismatched_children: Field = ArrowField::new(
            "a",
            DataType::Struct(Fields::from(vec![
                ArrowField::new("d", DataType::Int32, false),
                ArrowField::new("b", DataType::Int32, true),
            ])),
            true,
        )
        .try_into()
        .unwrap();
        assert_eq!(
            mismatched_children.explain_difference(&expected, &opts),
            Some("`a` had mismatched children, missing=[c] unexpected=[d]".to_string())
        );

        let reordered_children: Field = ArrowField::new(
            "a",
            DataType::Struct(Fields::from(vec![
                ArrowField::new("c", DataType::Int32, false),
                ArrowField::new("b", DataType::Int32, true),
            ])),
            true,
        )
        .try_into()
        .unwrap();
        assert_eq!(
            reordered_children.explain_difference(&expected, &opts),
            Some("`a` field order mismatch, expected [b, c] but was [c, b]".to_string())
        );

        let multiple_wrongs: Field = ArrowField::new(
            "c",
            DataType::Struct(Fields::from(vec![
                ArrowField::new("b", DataType::Int32, true),
                ArrowField::new("c", DataType::Float32, true),
            ])),
            true,
        )
        .try_into()
        .unwrap();
        assert_eq!(
            multiple_wrongs.explain_difference(&expected, &opts),
            Some(
                "expected name 'a' but name was 'c', `c.c` should have type int32 but type was float"
                    .to_string()
            )
        );

        let mut expected: Field = ArrowField::new(
            "a",
            DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
            true,
        )
        .try_into()
        .unwrap();
        let keys = UInt32Array::from_iter_values(vec![0, 1]);
        let values: ArrayRef = Arc::new(StringArray::from_iter_values([
            "a".to_string(),
            "b".to_string(),
        ]));
        let dictionary: ArrayRef = Arc::new(DictionaryArray::new(keys, values));
        expected.set_dictionary(&dictionary);

        let no_dictionary: Field = ArrowField::new(
            "a",
            DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
            true,
        )
        .try_into()
        .unwrap();

        // By default, do not compare dictionary
        assert_eq!(no_dictionary.explain_difference(&expected, &opts), None);

        let compare_dict = SchemaCompareOptions {
            compare_dictionary: true,
            ..Default::default()
        };
        assert_eq!(
            no_dictionary.explain_difference(&expected, &compare_dict),
            Some("dictionary for `a` did not match expected dictionary".to_string())
        );

        let metadata = HashMap::<_, _>::from_iter(vec![("foo".to_string(), "bar".to_string())]);
        let expected: Field = ArrowField::new("a", DataType::UInt32, true)
            .with_metadata(metadata)
            .try_into()
            .unwrap();

        let no_metadata: Field = ArrowField::new("a", DataType::UInt32, true)
            .try_into()
            .unwrap();

        // By default, do not compare metadata
        assert_eq!(no_metadata.explain_difference(&expected, &opts), None);

        let compare_metadata = SchemaCompareOptions {
            compare_metadata: true,
            ..Default::default()
        };
        assert_eq!(
            no_metadata.explain_difference(&expected, &compare_metadata),
            Some("metadata for `a` did not match expected metadata".to_string())
        );

        let mut expected: Field = ArrowField::new("a", DataType::UInt32, true)
            .try_into()
            .unwrap();
        let mut seed = 0;
        expected.set_id(-1, &mut seed);

        let no_id: Field = ArrowField::new("a", DataType::UInt32, true)
            .try_into()
            .unwrap();
        // Do not compare ids by default
        assert_eq!(no_id.explain_difference(&expected, &opts), None);

        let compare_ids = SchemaCompareOptions {
            compare_field_ids: true,
            ..Default::default()
        };
        assert_eq!(
            no_id.explain_difference(&expected, &compare_ids),
            Some("`a` should have id 0 but id was -1".to_string())
        );
    }
}
