//! Lance data types, [Schema] and [Field]

use std::collections::HashMap;
use std::fmt;
use std::fmt::Formatter;

use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema, TimeUnit};

use crate::encodings::Encoding;
use crate::format::pb;
use crate::{LanceError, Result};

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
    type Error = LanceError;

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
            DataType::Utf8 => "string".to_string(),
            DataType::Binary => "binary".to_string(),
            DataType::LargeUtf8 => "large_string".to_string(),
            DataType::LargeBinary => "large_binary".to_string(),
            DataType::Date32 => "date32:day".to_string(),
            DataType::Date64 => "date64:ms".to_string(),
            DataType::Time32(tu) => format!("time32:{}", timeunit_to_str(tu)),
            DataType::Time64(tu) => format!("time64:{}", timeunit_to_str(tu)),
            DataType::Timestamp(tu, _) => format!("timestamp:{}", timeunit_to_str(tu)),
            DataType::Struct(_) => "struct".to_string(),
            DataType::List(elem) => match elem.data_type() {
                DataType::Struct(_) => "list.struct".to_string(),
                _ => "list".to_string(),
            },
            DataType::FixedSizeList(dt, len) => format!(
                "fixed_size_list:{}:{}",
                LogicalType::try_from(dt.data_type())?.0,
                *len
            ),
            DataType::FixedSizeBinary(len) => format!("fixed_size_binary:{}", *len),
            _ => return Err(LanceError::Schema(format!("Unsupport data type: {:?}", dt))),
        };

        Ok(Self(type_str.to_string()))
    }
}

impl TryFrom<&LogicalType> for DataType {
    type Error = LanceError;

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
            _ => None,
        } {
            Ok(t)
        } else {
            let splits = lt.0.split(":").collect::<Vec<_>>();
            match splits[0] {
                "fixed_size_list" => {
                    if splits.len() != 3 {
                        Err(LanceError::Schema(format!("Unsupported logical type: {}", lt)))
                    } else {
                        let elem_type = (&LogicalType(splits[1].to_string())).try_into()?;
                        let size: i32 = splits[2].parse::<i32>().map_err(|e: _| LanceError::Schema(e.to_string()))?;
                        Ok(FixedSizeList(
                            Box::new(ArrowField::new("item", elem_type, true)),
                            size,
                        ))
                    }
                }
                "fixed_size_binary" => {
                    if splits.len() != 2 {
                        Err(LanceError::Schema(format!("Unsupported logical type: {}", lt)))
                    } else {
                        let size: i32 = splits[1].parse::<i32>().map_err(|e: _| LanceError::Schema(e.to_string()))?;
                        Ok(FixedSizeBinary(size))
                    }
                }
                _ => Err(LanceError::Schema(format!("Unsupported logical type: {}", lt))),
            }
        }
    }
}

fn is_numeric(data_type: &DataType) -> bool {
    use DataType::*;
    matches!(
        data_type,
        UInt8 | UInt16 | UInt32 | UInt64 | Int8 | Int16 | Int32 | Int64 | Float32 | Float64
    )
}

fn is_binary(data_type: &DataType) -> bool {
    use DataType::*;
    matches!(data_type, Binary | Utf8 | LargeBinary | LargeUtf8)
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
    encoding: Option<Encoding>,
    nullable: bool,

    children: Vec<Field>,
}

impl Field {
    /// Returns arrow data type.
    pub fn data_type(&self) -> DataType {
        match &self.logical_type {
            lt if lt.is_list() => DataType::List(Box::new(ArrowField::from(&self.children[0]))),
            lt if lt.is_struct() => {
                DataType::Struct(self.children.iter().map(ArrowField::from).collect())
            }
            lt => DataType::try_from(lt).unwrap(),
        }
    }

    pub fn child(&self, name: &str) -> Option<&Field> {
        self.children.iter().find(|f| f.name == name)
    }

    pub fn child_mut(&mut self, name: &str) -> Option<&mut Field> {
        self.children.iter_mut().find(|f| f.name == name)
    }

    fn project(&self, path_components: &[&str]) -> Result<Field> {
        let mut f = Field {
            name: self.name.clone(),
            id: self.id,
            parent_id: self.parent_id,
            logical_type: self.logical_type.clone(),
            extension_name: self.extension_name.clone(),
            encoding: self.encoding.clone(),
            nullable: self.nullable,
            children: vec![],
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
    fn merge(&mut self, other: &Field) -> Result<()> {
        for other_child in other.children.as_slice() {
            if let Some(field) = self.child_mut(&other_child.name) {
                field.merge(other_child)?;
            } else {
                self.children.push(other_child.clone());
            }
        }
        Ok(())
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
    type Error = LanceError;

    fn try_from(field: &ArrowField) -> Result<Self> {
        let children = match field.data_type() {
            DataType::Struct(children) => {
                children.iter().map(Self::try_from).collect::<Result<_>>()?
            }
            DataType::List(item) => vec![Self::try_from(item.as_ref())?],
            _ => vec![],
        };
        Ok(Self {
            id: -1,
            parent_id: -1,
            name: field.name().clone(),
            logical_type: LogicalType::try_from(field.data_type())?,
            encoding: match field.data_type() {
                dt if is_numeric(dt) => Some(Encoding::Plain),
                dt if is_binary(dt) => Some(Encoding::VarBinary),
                DataType::Dictionary(_, _) => Some(Encoding::Dictionary),
                _ => None,
            },
            extension_name: "".to_string(),
            nullable: field.is_nullable(),
            children,
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
            dictionary: None,
            extension_name: field.extension_name.clone(),
            r#type: 0,
        }
    }
}

/// Lance Schema.
#[derive(Default, Debug, PartialEq)]
pub struct Schema {
    pub fields: Vec<Field>,
    pub metadata: HashMap<String, String>,
}

impl Schema {
    /// Project the columns over the schema.
    ///
    ///
    pub fn project(&self, columns: &[&str]) -> Result<Schema> {
        let mut candidates: Vec<Field> = vec![];
        for col in columns {
            let split = (*col).split('.').collect::<Vec<_>>();
            let first = split[0];
            if let Some(field) = self.field(first) {
                let projected_field = field.project(&split[1..])?;
                if let Some(candidate_field) =
                    candidates.iter_mut().filter(|f| f.name == first).next()
                {
                    candidate_field.merge(&projected_field)?;
                } else {
                    candidates.push(projected_field)
                }
            } else {
                return Err(LanceError::Schema(format!("Column {} does not exist", col)));
            }
        }

        Ok(Schema {
            fields: candidates,
            metadata: self.metadata.clone(),
        })
    }

    fn field(&self, name: &str) -> Option<&Field> {
        self.fields.iter().filter(|f| f.name == name).next()
    }
}

impl fmt::Display for Schema {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for field in self.fields.iter() {
            writeln!(f, "{}", field)?
        }
        Ok(())
    }
}

/// Convert `arrow2::datatype::Schema` to Lance
impl TryFrom<&ArrowSchema> for Schema {
    type Error = LanceError;

    fn try_from(schema: &ArrowSchema) -> Result<Self> {
        Ok(Self {
            fields: schema
                .fields
                .iter()
                .map(Field::try_from)
                .collect::<Result<_>>()?,
            metadata: schema.metadata.clone(),
        })
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
        Self {
            fields: fields.iter().map(Field::from).collect(),
            metadata: HashMap::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_schema::{Field as ArrowField, TimeUnit};

    use super::*;

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
        assert_eq!(projected, Schema::try_from(&expected_arrow_schema).unwrap());
    }
}
