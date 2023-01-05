//! Lance data types

use std::collections::HashMap;
use std::fmt;
use std::fmt::Formatter;

use arrow_schema::{ArrowError, DataType, Field as ArrowField, Schema as ArrowSchema};

use crate::encodings::Encoding;
use crate::format::pb;

/// LogicalType is a string presentation of arrow type.
/// to be serialized into protobuf.
#[derive(Debug)]
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

impl TryFrom<&DataType> for LogicalType {
    type Error = String;

    fn try_from(dt: &DataType) -> Result<Self, Self::Error> {
        let type_str = match dt {
            DataType::Null => Ok("null"),
            DataType::Boolean => Ok("bool"),
            DataType::Int8 => Ok("int8"),
            DataType::UInt8 => Ok("uint8"),
            DataType::Int16 => Ok("int16"),
            DataType::UInt16 => Ok("uint16"),
            DataType::Int32 => Ok("int32"),
            DataType::UInt32 => Ok("uint32"),
            DataType::Int64 => Ok("int64"),
            DataType::UInt64 => Ok("uint64"),
            DataType::Float16 => Ok("halffloat"),
            DataType::Float32 => Ok("float"),
            DataType::Float64 => Ok("double"),
            DataType::Utf8 => Ok("string"),
            DataType::Binary => Ok("binary"),
            DataType::LargeUtf8 => Ok("large_string"),
            DataType::LargeBinary => Ok("large_binary"),
            DataType::Date32 => Ok("date32:day"),
            DataType::Date64 => Ok("date64:ms"),
            DataType::Struct(_) => Ok("struct"),
            _ => Err(format!("Unsupport data type: {:?}", dt)),
        }?;

        Ok(Self(type_str.to_string()))
    }
}

impl TryFrom<&LogicalType> for DataType {
    type Error = String;

    fn try_from(lt: &LogicalType) -> Result<Self, Self::Error> {
        use DataType::*;
        match lt.0.as_str() {
            "null" => Ok(Null),
            "bool" => Ok(Boolean),
            "int8" => Ok(Int8),
            "uint8" => Ok(UInt8),
            "int16" => Ok(Int16),
            "uint16" => Ok(UInt16),
            "int32" => Ok(Int32),
            "uint32" => Ok(UInt32),
            "int64" => Ok(Int64),
            "uint64" => Ok(UInt64),
            "halffloat" => Ok(Float16),
            "float" => Ok(Float32),
            "double" => Ok(Float64),
            "string" => Ok(Utf8),
            "binary" => Ok(Binary),
            "large_string" => Ok(LargeUtf8),
            "large_binary" => Ok(LargeBinary),
            "date32:day" => Ok(Date32),
            "date64:ms" => Ok(Date64),
            _ => Err(format!("Unsupported type, {}", lt.0.as_str())),
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
#[derive(Debug)]
pub struct Field {
    pub name: String,
    pub id: i32,
    pub parent_id: i32,
    pub logical_type: LogicalType,
    pub extension_name: String,
    pub encoding: Option<Encoding>,
    pub nullable: bool,

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
    type Error = ArrowError;

    fn try_from(field: &ArrowField) -> Result<Self, ArrowError> {
        let children = match field.data_type() {
            DataType::Struct(children) => children
                .iter()
                .map(Self::try_from)
                .collect::<Result<_, _>>()?,
            DataType::List(item) => vec![Self::try_from(item.as_ref())?],
            _ => vec![],
        };
        Ok(Self {
            id: -1,
            parent_id: -1,
            name: field.name().clone(),
            logical_type: LogicalType::try_from(field.data_type())
                .map_err(ArrowError::SchemaError)?,
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
#[derive(Default, Debug)]
pub struct Schema {
    pub fields: Vec<Field>,
    pub metadata: HashMap<String, String>,
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
    type Error = ArrowError;

    fn try_from(schema: &ArrowSchema) -> Result<Self, ArrowError> {
        Ok(Self {
            fields: schema
                .fields
                .iter()
                .map(Field::try_from)
                .collect::<Result<_, _>>()?,
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
    use arrow_schema::Field as ArrowField;

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
        ] {
            let arrow_field = ArrowField::new(name, data_type.clone(), true);
            let field = Field::try_from(&arrow_field).unwrap();
            assert_eq!(field.name, name);
            assert_eq!(field.data_type(), data_type);
            assert_eq!(ArrowField::try_from(&field).unwrap(), arrow_field);
        }
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
}
