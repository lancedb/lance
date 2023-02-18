//! Lance data types, [Schema] and [Field]

use std::cmp::max;
use std::collections::HashMap;
use std::fmt::Formatter;
use std::fmt::{self};

use arrow_array::types::{
    Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
};
use arrow_array::{cast::as_dictionary_array, ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema, TimeUnit};
use async_recursion::async_recursion;

use crate::arrow::DataTypeExt;
use crate::encodings::Encoding;
use crate::format::pb;
use crate::io::object_reader::{read_binary_array, read_fixed_stride_array, ObjectReader};
use crate::{Error, Result};

/// LogicalType is a string presentation of arrow type.
/// to be serialized into protobuf.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalType(String);

impl fmt::Display for LogicalType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl LogicalType {
    fn is_list(&self) -> bool {
        self.0 == "list" || self.0 == "list.struct"
    }

    fn is_large_list(&self) -> bool {
        self.0 == "large_list" || self.0 == "large_list.struct"
    }

    fn is_struct(&self) -> bool {
        self.0 == "struct"
    }
}

impl From<&str> for LogicalType {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

fn timeunit_to_str(unit: &TimeUnit) -> &'static str {
    match unit {
        TimeUnit::Second => "s",
        TimeUnit::Millisecond => "ms",
        TimeUnit::Microsecond => "us",
        TimeUnit::Nanosecond => "ns",
    }
}

impl TryFrom<&DataType> for LogicalType {
    type Error = Error;

    fn try_from(dt: &DataType) -> Result<Self> {
        let type_str = match dt {
            DataType::Null => "null".to_string(),
            DataType::Boolean => "bool".to_string(),
            DataType::Int8 => "int8".to_string(),
            DataType::UInt8 => "uint8".to_string(),
            DataType::Int16 => "int16".to_string(),
            DataType::UInt16 => "uint16".to_string(),
            DataType::Int32 => "int32".to_string(),
            DataType::UInt32 => "uint32".to_string(),
            DataType::Int64 => "int64".to_string(),
            DataType::UInt64 => "uint64".to_string(),
            DataType::Float16 => "halffloat".to_string(),
            DataType::Float32 => "float".to_string(),
            DataType::Float64 => "double".to_string(),
            DataType::Decimal128(precision, scale) => format!("decimal:128:{precision}:{scale}"),
            DataType::Decimal256(precision, scale) => format!("decimal:256:{precision}:{scale}"),
            DataType::Utf8 => "string".to_string(),
            DataType::Binary => "binary".to_string(),
            DataType::LargeUtf8 => "large_string".to_string(),
            DataType::LargeBinary => "large_binary".to_string(),
            DataType::Date32 => "date32:day".to_string(),
            DataType::Date64 => "date64:ms".to_string(),
            DataType::Time32(tu) => format!("time32:{}", timeunit_to_str(tu)),
            DataType::Time64(tu) => format!("time64:{}", timeunit_to_str(tu)),
            DataType::Timestamp(tu, _) => format!("timestamp:{}", timeunit_to_str(tu)),
            DataType::Duration(tu) => format!("duration:{}", timeunit_to_str(tu)),
            DataType::Struct(_) => "struct".to_string(),
            DataType::Dictionary(key_type, value_type) => {
                format!(
                    "dict:{}:{}:{}",
                    Self::try_from(value_type.as_ref())?.0,
                    Self::try_from(key_type.as_ref())?.0,
                    // Arrow C++ Dictionary has "ordered:bool" field, but it does not exist in `arrow-rs`.
                    false
                )
            }
            DataType::List(elem) => match elem.data_type() {
                DataType::Struct(_) => "list.struct".to_string(),
                _ => "list".to_string(),
            },
            DataType::LargeList(elem) => match elem.data_type() {
                DataType::Struct(_) => "large_list.struct".to_string(),
                _ => "large_list".to_string(),
            },
            DataType::FixedSizeList(dt, len) => format!(
                "fixed_size_list:{}:{}",
                Self::try_from(dt.data_type())?.0,
                *len
            ),
            DataType::FixedSizeBinary(len) => format!("fixed_size_binary:{}", *len),
            _ => return Err(Error::Schema(format!("Unsupport data type: {:?}", dt))),
        };

        Ok(Self(type_str))
    }
}

impl TryFrom<&LogicalType> for DataType {
    type Error = Error;

    fn try_from(lt: &LogicalType) -> Result<Self> {
        use DataType::*;
        if let Some(t) = match lt.0.as_str() {
            "null" => Some(Null),
            "bool" => Some(Boolean),
            "int8" => Some(Int8),
            "uint8" => Some(UInt8),
            "int16" => Some(Int16),
            "uint16" => Some(UInt16),
            "int32" => Some(Int32),
            "uint32" => Some(UInt32),
            "int64" => Some(Int64),
            "uint64" => Some(UInt64),
            "halffloat" => Some(Float16),
            "float" => Some(Float32),
            "double" => Some(Float64),
            "string" => Some(Utf8),
            "binary" => Some(Binary),
            "large_string" => Some(LargeUtf8),
            "large_binary" => Some(LargeBinary),
            "date32:day" => Some(Date32),
            "date64:ms" => Some(Date64),
            "time32:s" => Some(Time32(TimeUnit::Second)),
            "time32:ms" => Some(Time32(TimeUnit::Millisecond)),
            "time64:us" => Some(Time64(TimeUnit::Microsecond)),
            "time64:ns" => Some(Time64(TimeUnit::Nanosecond)),
            "timestamp:s" => Some(Timestamp(TimeUnit::Second, None)),
            "timestamp:ms" => Some(Timestamp(TimeUnit::Millisecond, None)),
            "timestamp:us" => Some(Timestamp(TimeUnit::Microsecond, None)),
            "timestamp:ns" => Some(Timestamp(TimeUnit::Nanosecond, None)),
            "duration:s" => Some(Duration(TimeUnit::Second)),
            "duration:ms" => Some(Duration(TimeUnit::Millisecond)),
            "duration:us" => Some(Duration(TimeUnit::Microsecond)),
            "duration:ns" => Some(Duration(TimeUnit::Nanosecond)),
            _ => None,
        } {
            Ok(t)
        } else {
            let splits = lt.0.split(':').collect::<Vec<_>>();
            match splits[0] {
                "fixed_size_list" => {
                    if splits.len() != 3 {
                        Err(Error::Schema(format!("Unsupported logical type: {}", lt)))
                    } else {
                        let elem_type = (&LogicalType(splits[1].to_string())).try_into()?;
                        let size: i32 = splits[2]
                            .parse::<i32>()
                            .map_err(|e: _| Error::Schema(e.to_string()))?;
                        Ok(FixedSizeList(
                            Box::new(ArrowField::new("item", elem_type, true)),
                            size,
                        ))
                    }
                }
                "fixed_size_binary" => {
                    if splits.len() != 2 {
                        Err(Error::Schema(format!("Unsupported logical type: {}", lt)))
                    } else {
                        let size: i32 = splits[1]
                            .parse::<i32>()
                            .map_err(|e: _| Error::Schema(e.to_string()))?;
                        Ok(FixedSizeBinary(size))
                    }
                }
                "dict" => {
                    if splits.len() != 4 {
                        Err(Error::Schema(format!("Unsupport dictionary type: {}", lt)))
                    } else {
                        let value_type: Self = (&LogicalType::from(splits[1])).try_into()?;
                        let index_type: Self = (&LogicalType::from(splits[2])).try_into()?;
                        Ok(Dictionary(Box::new(index_type), Box::new(value_type)))
                    }
                }
                "decimal" => {
                    if splits.len() != 4 {
                        Err(Error::Schema(format!("Unsupport decimal type: {}", lt)))
                    } else {
                        let bits: i16 = splits[1]
                            .parse::<i16>()
                            .map_err(|err| Error::Schema(err.to_string()))?;
                        let precision: u8 = splits[2]
                            .parse::<u8>()
                            .map_err(|err| Error::Schema(err.to_string()))?;
                        let scale: i8 = splits[3]
                            .parse::<i8>()
                            .map_err(|err| Error::Schema(err.to_string()))?;

                        if bits == 128 {
                            Ok(Decimal128(precision, scale))
                        } else if bits == 256 {
                            Ok(Decimal256(precision, scale))
                        } else {
                            Err(Error::Schema(format!(
                                "Only Decimal128 and Decimal256 is supported. Found {bits}"
                            )))
                        }
                    }
                }
                _ => Err(Error::Schema(format!("Unsupported logical type: {}", lt))),
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Dictionary {
    pub(crate) offset: usize,

    pub(crate) length: usize,

    pub(crate) values: Option<ArrayRef>,
}

impl From<&pb::Dictionary> for Dictionary {
    fn from(proto: &pb::Dictionary) -> Self {
        Self {
            offset: proto.offset as usize,
            length: proto.length as usize,
            values: None,
        }
    }
}

impl From<&Dictionary> for pb::Dictionary {
    fn from(d: &Dictionary) -> Self {
        Self {
            offset: d.offset as i64,
            length: d.length as i64,
        }
    }
}

/// Lance Schema Field
///
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub name: String,
    pub id: i32,
    parent_id: i32,
    logical_type: LogicalType,
    extension_name: String,
    pub(crate) encoding: Option<Encoding>,
    pub nullable: bool,

    pub children: Vec<Field>,

    /// Dictionary value array if this field is dictionary.
    pub dictionary: Option<Dictionary>,
}

impl Field {
    /// Returns arrow data type.
    pub fn data_type(&self) -> DataType {
        match &self.logical_type {
            lt if lt.is_list() => DataType::List(Box::new(ArrowField::from(&self.children[0]))),
            lt if lt.is_large_list() => {
                DataType::LargeList(Box::new(ArrowField::from(&self.children[0])))
            }
            lt if lt.is_struct() => {
                DataType::Struct(self.children.iter().map(ArrowField::from).collect())
            }
            lt => DataType::try_from(lt).unwrap(),
        }
    }

    pub fn child(&self, name: &str) -> Option<&Self> {
        self.children.iter().find(|f| f.name == name)
    }

    pub fn child_mut(&mut self, name: &str) -> Option<&mut Self> {
        self.children.iter_mut().find(|f| f.name == name)
    }

    /// Attach the Dictionary's value array, so that we can later serialize
    /// the dictionary to the manifest.
    pub(crate) fn set_dictionary_values(&mut self, arr: &ArrayRef) {
        assert!(self.data_type().is_dictionary());
        self.dictionary = Some(Dictionary {
            offset: 0,
            length: 0,
            values: Some(arr.clone()),
        });
    }

    fn set_dictionary(&mut self, arr: &ArrayRef) {
        let data_type = self.data_type();
        match data_type {
            DataType::Dictionary(key_type, _) => match key_type.as_ref() {
                DataType::Int8 => {
                    self.set_dictionary_values(as_dictionary_array::<Int8Type>(arr).values())
                }
                DataType::Int16 => {
                    self.set_dictionary_values(as_dictionary_array::<Int16Type>(arr).values())
                }
                DataType::Int32 => {
                    self.set_dictionary_values(as_dictionary_array::<Int32Type>(arr).values())
                }
                DataType::Int64 => {
                    self.set_dictionary_values(as_dictionary_array::<Int64Type>(arr).values())
                }
                DataType::UInt8 => {
                    self.set_dictionary_values(as_dictionary_array::<UInt8Type>(arr).values())
                }
                DataType::UInt16 => {
                    self.set_dictionary_values(as_dictionary_array::<UInt16Type>(arr).values())
                }
                DataType::UInt32 => {
                    self.set_dictionary_values(as_dictionary_array::<UInt32Type>(arr).values())
                }
                DataType::UInt64 => {
                    self.set_dictionary_values(as_dictionary_array::<UInt64Type>(arr).values())
                }
                _ => {
                    panic!("Unsupported dictionary key type: {}", key_type);
                }
            },
            _ => {
                // Add nested struct support.
            }
        }
    }

    fn project(&self, path_components: &[&str]) -> Result<Self> {
        let mut f = Self {
            name: self.name.clone(),
            id: self.id,
            parent_id: self.parent_id,
            logical_type: self.logical_type.clone(),
            extension_name: self.extension_name.clone(),
            encoding: self.encoding.clone(),
            nullable: self.nullable,
            children: vec![],
            dictionary: self.dictionary.clone(),
        };
        if path_components.is_empty() {
            // Project stops here, copy all the remaining children.
            f.children = self.children.clone()
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

    /// Merge the children of other field into this one.
    fn merge(&mut self, other: &Self) -> Result<()> {
        for other_child in other.children.as_slice() {
            if let Some(field) = self.child_mut(&other_child.name) {
                field.merge(other_child)?;
            } else {
                self.children.push(other_child.clone());
            }
        }
        Ok(())
    }

    // Get the max field id of itself and all children.
    fn max_id(&self) -> i32 {
        max(
            self.id,
            self.children.iter().map(|c| c.max_id()).max().unwrap_or(-1),
        )
    }

    /// Recursively set field ID and parent ID for this field and all its children.
    fn set_id(&mut self, parent_id: i32, id_seed: &mut i32) {
        self.parent_id = parent_id;
        if self.id < 0 {
            self.id = *id_seed;
            *id_seed += 1;
        }
        self.children
            .iter_mut()
            .for_each(|f| f.set_id(self.id, id_seed));
    }

    // Find any nested child with a specific field id
    fn mut_field_by_id(&mut self, id: i32) -> Option<&mut Self> {
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

    #[async_recursion]
    async fn load_dictionary<'a>(&mut self, reader: &dyn ObjectReader) -> Result<()> {
        if let DataType::Dictionary(_, value_type) = self.data_type() {
            assert!(self.dictionary.is_some());
            if let Some(dict_info) = self.dictionary.as_mut() {
                use DataType::*;
                match value_type.as_ref() {
                    Utf8 | Binary => {
                        dict_info.values = Some(
                            read_binary_array(
                                reader,
                                value_type.as_ref(),
                                false,
                                dict_info.offset,
                                dict_info.length,
                                ..,
                            )
                            .await?,
                        );
                    }
                    Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64 => {
                        dict_info.values = Some(
                            read_fixed_stride_array(
                                reader,
                                value_type.as_ref(),
                                dict_info.offset,
                                dict_info.length,
                                ..,
                            )
                            .await?,
                        );
                    }
                    _ => {
                        return Err(Error::Schema(format!(
                            "Does not support {} as dictionary value type",
                            value_type
                        )));
                    }
                }
            } else {
                panic!("Should not reach here: dictionary field does not load dictionary info")
            }
            Ok(())
        } else {
            for child in self.children.as_mut_slice() {
                child.load_dictionary(reader).await?;
            }
            Ok(())
        }
    }
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Field(id={}, name={}, type={})",
            self.id, self.name, self.logical_type.0,
        )
    }
}

impl TryFrom<&ArrowField> for Field {
    type Error = Error;

    fn try_from(field: &ArrowField) -> Result<Self> {
        let children = match field.data_type() {
            DataType::Struct(children) => {
                children.iter().map(Self::try_from).collect::<Result<_>>()?
            }
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
            extension_name: "".to_string(),
            nullable: field.is_nullable(),
            children,
            dictionary: None,
        })
    }
}

impl From<&Field> for ArrowField {
    fn from(field: &Field) -> Self {
        Self::new(&field.name, field.data_type(), field.nullable)
    }
}

impl From<&pb::Field> for Field {
    fn from(field: &pb::Field) -> Self {
        Self {
            name: field.name.clone(),
            id: field.id,
            parent_id: field.parent_id,
            logical_type: LogicalType(field.logical_type.clone()),
            extension_name: field.extension_name.clone(),
            encoding: match field.encoding {
                1 => Some(Encoding::Plain),
                2 => Some(Encoding::VarBinary),
                3 => Some(Encoding::Dictionary),
                4 => Some(Encoding::RLE),
                _ => None,
            },
            nullable: field.nullable,
            children: vec![],
            dictionary: field.dictionary.as_ref().map(Dictionary::from),
        }
    }
}

impl From<&Field> for pb::Field {
    fn from(field: &Field) -> Self {
        Self {
            id: field.id,
            parent_id: field.parent_id,
            name: field.name.clone(),
            logical_type: field.logical_type.0.clone(),
            encoding: match field.encoding {
                Some(Encoding::Plain) => 1,
                Some(Encoding::VarBinary) => 2,
                Some(Encoding::Dictionary) => 3,
                Some(Encoding::RLE) => 4,
                _ => 0,
            },
            nullable: field.nullable,
            dictionary: field.dictionary.as_ref().map(pb::Dictionary::from),
            extension_name: field.extension_name.clone(),
            r#type: 0,
        }
    }
}

impl From<&Field> for Vec<pb::Field> {
    fn from(field: &Field) -> Self {
        let mut protos = vec![pb::Field::from(field)];
        protos.extend(field.children.iter().flat_map(Self::from));
        protos
    }
}

/// Lance Schema.
#[derive(Default, Debug, PartialEq, Clone)]
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
    pub fn project(&self, columns: &[&str]) -> Result<Self> {
        let mut candidates: Vec<Field> = vec![];
        for col in columns {
            let split = (*col).split('.').collect::<Vec<_>>();
            let first = split[0];
            if let Some(field) = self.field(first) {
                let projected_field = field.project(&split[1..])?;
                if let Some(candidate_field) = candidates.iter_mut().find(|f| f.name == first) {
                    candidate_field.merge(&projected_field)?;
                } else {
                    candidates.push(projected_field)
                }
            } else {
                return Err(Error::Schema(format!("Column {} does not exist", col)));
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

    pub fn field(&self, name: &str) -> Option<&Field> {
        self.fields.iter().find(|f| f.name == name)
    }

    pub(crate) fn field_id(&self, column: &str) -> Result<i32> {
        self.field(column)
            .map(|f| f.id)
            .ok_or_else(|| Error::Schema("Vector column not in schema".to_string()))
    }

    /// Recursively collect all the field IDs,
    pub(crate) fn field_ids(&self) -> Vec<i32> {
        // TODO: make a tree travesal iterator.

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
            let column = batch.column_by_name(&field.name).ok_or_else(|| {
                Error::Schema(format!(
                    "column '{}' does not exist in the record batch",
                    field.name
                ))
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

    pub fn merge(&self, other: &Self) -> Self {
        let mut fields = self.fields.clone();
        for field in other.fields.as_slice() {
            if !fields.iter().any(|f| f.name == field.name) {
                fields.push(field.clone());
            }
        }
        let mut metadata = other.metadata.clone();
        self.metadata.iter().for_each(|(k, v)| {
            metadata.insert(k.to_string(), v.to_string());
        });
        Self { fields, metadata }
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
                .map(Field::try_from)
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

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_schema::{Field as ArrowField, TimeUnit};

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
            ("timestamp:s", DataType::Timestamp(TimeUnit::Second, None)),
            (
                "timestamp:ms",
                DataType::Timestamp(TimeUnit::Millisecond, None),
            ),
            (
                "timestamp:us",
                DataType::Timestamp(TimeUnit::Microsecond, None),
            ),
            (
                "timestamp:ns",
                DataType::Timestamp(TimeUnit::Nanosecond, None),
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
                    Box::new(ArrowField::new("item", DataType::Int32, true)),
                    10,
                ),
            ),
        ] {
            let arrow_field = ArrowField::new(name, data_type.clone(), true);
            let field = Field::try_from(&arrow_field).unwrap();
            assert_eq!(field.name, name);
            assert_eq!(field.data_type(), data_type);
            assert_eq!(ArrowField::try_from(&field).unwrap(), arrow_field);
        }
    }

    #[test]
    fn test_nested_types() {
        assert_eq!(
            LogicalType::try_from(&DataType::List(Box::new(ArrowField::new(
                "item",
                DataType::Binary,
                false
            ))))
            .unwrap()
            .0,
            "list"
        );
        assert_eq!(
            LogicalType::try_from(&DataType::List(Box::new(ArrowField::new(
                "item",
                DataType::Struct(vec![]),
                false
            ))))
            .unwrap()
            .0,
            "list.struct"
        );
        assert_eq!(
            LogicalType::try_from(&DataType::Struct(vec![ArrowField::new(
                "item",
                DataType::Binary,
                false
            )]))
            .unwrap()
            .0,
            "struct"
        );
    }

    #[test]
    fn struct_field() {
        let arrow_field = ArrowField::new(
            "struct",
            DataType::Struct(vec![ArrowField::new("a", DataType::Int32, true)]),
            false,
        );
        let field = Field::try_from(&arrow_field).unwrap();
        assert_eq!(field.name, "struct");
        assert_eq!(&field.data_type(), arrow_field.data_type());
        assert_eq!(ArrowField::try_from(&field).unwrap(), arrow_field);
    }

    #[test]
    fn test_schema_projection() {
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new(
                "b",
                DataType::Struct(vec![
                    ArrowField::new("f1", DataType::Utf8, true),
                    ArrowField::new("f2", DataType::Boolean, false),
                    ArrowField::new("f3", DataType::Float32, false),
                ]),
                true,
            ),
            ArrowField::new("c", DataType::Float64, false),
        ]);
        let schema = Schema::try_from(&arrow_schema).unwrap();
        let projected = schema.project(&["b.f1", "b.f3", "c"]).unwrap();

        let expected_arrow_schema = ArrowSchema::new(vec![
            ArrowField::new(
                "b",
                DataType::Struct(vec![
                    ArrowField::new("f1", DataType::Utf8, true),
                    ArrowField::new("f3", DataType::Float32, false),
                ]),
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
                DataType::Struct(vec![
                    ArrowField::new("f1", DataType::Utf8, true),
                    ArrowField::new("f2", DataType::Boolean, false),
                    ArrowField::new("f3", DataType::Float32, false),
                ]),
                true,
            ),
            ArrowField::new("c", DataType::Float64, false),
        ]);
        let schema = Schema::try_from(&arrow_schema).unwrap();
        let projected = schema.project_by_ids(&[1, 2, 4, 5]).unwrap();

        let expected_arrow_schema = ArrowSchema::new(vec![
            ArrowField::new(
                "b",
                DataType::Struct(vec![
                    ArrowField::new("f1", DataType::Utf8, true),
                    ArrowField::new("f3", DataType::Float32, false),
                ]),
                true,
            ),
            ArrowField::new("c", DataType::Float64, false),
        ]);
        assert_eq!(ArrowSchema::from(&projected), expected_arrow_schema);
    }

    #[test]
    fn test_schema_set_ids() {
        let arrow_schema = ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new(
                "b",
                DataType::Struct(vec![
                    ArrowField::new("f1", DataType::Utf8, true),
                    ArrowField::new("f2", DataType::Boolean, false),
                    ArrowField::new("f3", DataType::Float32, false),
                ]),
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
}
