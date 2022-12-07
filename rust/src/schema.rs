//  Copyright 2022 Lance Authors
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

//! Lance Dataset Schema

use std::fmt;
use std::fmt::Debug;
use std::io::{Read, Seek};

use arrow2::datatypes::PhysicalType::Primitive;
use arrow2::datatypes::{DataType, PrimitiveType, TimeUnit};
use arrow2::types::{days_ms, f16, i256, months_days_ns};
use std::string::ToString;

use crate::encodings::plain::PlainDecoder;
use crate::encodings::{Decoder, Encoding};
use crate::format::pb;
use crate::page_table::PageInfo;

/// Lance Field.
///
/// Metadata of a column.
#[derive(Debug, Clone)]
pub struct Field {
    pub id: i32,
    pub parent_id: i32,
    pub name: String,
    pub logical_type: String,
    pub extension_name: String,
    pub encoding: Option<Encoding>,
    node_type: i32,

    children: Vec<Field>,
}

impl Field {
    pub fn new(field: &arrow2::datatypes::Field) -> Field {
        Field {
            id: -1,
            parent_id: -1,
            name: field.name.clone(),
            logical_type: field.data_type().to_logical_type().type_str(),
            extension_name: String::new(),
            encoding: match field.data_type() {
                t if t.is_numeric() => Some(Encoding::Plain),
                DataType::Binary | DataType::Utf8 | DataType::LargeBinary | DataType::LargeUtf8 => {
                    Some(Encoding::VarBinary)
                }
                DataType::Dictionary(_, _, _) => Some(Encoding::Dictionary),
                _ => None,
            },
            node_type: 0,
            children: vec![],
        }
        // TODO Add subfields.
    }

    pub fn fields(&self) -> &Vec<Field> {
        return &self.children;
    }

    pub fn get_decoder<'a, R: Read + Seek>(
        &'a self,
        reader: &'a mut R,
        page_info: PageInfo,
    ) -> Box<dyn Decoder + '_> {
        //Field::GetDecoder

        match self.data_type().to_physical_type() {
            Primitive(t) => match t {
                PrimitiveType::Int8 => {
                    return Box::new(PlainDecoder::<R, i8>::new(
                        reader,
                        page_info.position as u64,
                        page_info.length,
                    ));
                }
                PrimitiveType::Int16 => {
                    return Box::new(PlainDecoder::<R, i16>::new(
                        reader,
                        page_info.position as u64,
                        page_info.length,
                    ));
                }
                PrimitiveType::Int32 => {
                    return Box::new(PlainDecoder::<R, i32>::new(
                        reader,
                        page_info.position as u64,
                        page_info.length,
                    ));
                }
                PrimitiveType::Int64 => {
                    return Box::new(PlainDecoder::<R, i64>::new(
                        reader,
                        page_info.position as u64,
                        page_info.length,
                    ));
                }
                PrimitiveType::Int128 => {
                    return Box::new(PlainDecoder::<R, i128>::new(
                        reader,
                        page_info.position as u64,
                        page_info.length,
                    ));
                }
                PrimitiveType::Int256 => {
                    return Box::new(PlainDecoder::<R, i256>::new(
                        reader,
                        page_info.position as u64,
                        page_info.length,
                    ));
                }
                PrimitiveType::UInt8 => {
                    return Box::new(PlainDecoder::<R, u8>::new(
                        reader,
                        page_info.position as u64,
                        page_info.length,
                    ));
                }
                PrimitiveType::UInt16 => {
                    return Box::new(PlainDecoder::<R, u16>::new(
                        reader,
                        page_info.position as u64,
                        page_info.length,
                    ));
                }
                PrimitiveType::UInt32 => {
                    return Box::new(PlainDecoder::<R, u32>::new(
                        reader,
                        page_info.position as u64,
                        page_info.length,
                    ));
                }
                PrimitiveType::UInt64 => {
                    return Box::new(PlainDecoder::<R, u64>::new(
                        reader,
                        page_info.position as u64,
                        page_info.length,
                    ));
                }
                PrimitiveType::Float16 => {
                    return Box::new(PlainDecoder::<R, f16>::new(
                        reader,
                        page_info.position as u64,
                        page_info.length,
                    ));
                }
                PrimitiveType::Float32 => {
                    return Box::new(PlainDecoder::<R, f32>::new(
                        reader,
                        page_info.position as u64,
                        page_info.length,
                    ));
                }
                PrimitiveType::Float64 => {
                    return Box::new(PlainDecoder::<R, f64>::new(
                        reader,
                        page_info.position as u64,
                        page_info.length,
                    ));
                }
                PrimitiveType::DaysMs => {
                    return Box::new(PlainDecoder::<R, days_ms>::new(
                        reader,
                        page_info.position as u64,
                        page_info.length,
                    ));
                }
                PrimitiveType::MonthDayNano => {
                    return Box::new(PlainDecoder::<R, months_days_ns>::new(
                        reader,
                        page_info.position as u64,
                        page_info.length,
                    ));
                }
                _ => todo!(),
            },
            _ => todo!(),
        }

        todo!()
    }

    /// Build Field from protobuf.
    pub fn from_proto(pb: &pb::Field) -> Field {
        Field {
            id: pb.id,
            parent_id: pb.parent_id,
            name: pb.name.to_string(),
            logical_type: pb.logical_type.to_string(),
            extension_name: pb.extension_name.to_string(),
            encoding: match pb.encoding {
                1 => Some(Encoding::Plain),
                2 => Some(Encoding::VarBinary),
                3 => Some(Encoding::Dictionary),
                _ => None,
            },
            node_type: pb.r#type,

            children: vec![],
        }
    }

    /// Return Arrow Data Type.
    pub fn data_type(&self) -> DataType {
        match self.logical_type.as_str() {
            "bool" => DataType::Boolean,
            "uint8" => DataType::UInt8,
            "int8" => DataType::Int8,
            "uint16" => DataType::UInt16,
            "int16" => DataType::Int16,
            "uint32" => DataType::UInt32,
            "int32" => DataType::Int32,
            "uint64" => DataType::UInt64,
            "int64" => DataType::Int64,
            "halffloat" => DataType::Float16,
            "float" => DataType::Float32,
            "double" => DataType::Float64,
            "binary" => DataType::Binary,
            "string" => DataType::Utf8,
            x => DataType::Extension(
                "not_supported_yet".to_string(),
                Box::new(DataType::Binary),
                Some(x.to_string()),
            ),
        }
    }

    /// Return Arrow Data Type name.
    pub fn type_str(t: &DataType) -> String {
        match t {
            DataType::Boolean => "bool".to_string(),
            DataType::UInt8 => "uint8".to_string(),
            DataType::Int8 => "int8".to_string(),
            DataType::UInt16 => "uint16".to_string(),
            DataType::Int16 => "int16".to_string(),
            DataType::UInt32 => "uint32".to_string(),
            DataType::Int32 => "int32".to_string(),
            DataType::UInt64 => "uint64".to_string(),
            DataType::Int64 => "int64".to_string(),
            DataType::Float16 => "halffloat".to_string(),
            DataType::Float32 => "float".to_string(),
            DataType::Float64 => "double".to_string(),
            DataType::Date32 => "date32:day".to_string(),
            DataType::Date64 => "date64:ms".to_string(),
            DataType::Time32(unit) => format!("time32:{}", to_str(unit)),
            DataType::Time64(unit) => format!("time64:{}", to_str(unit)),
            DataType::Timestamp(unit, _) => format!("timestamp:{}", to_str(unit)),
            DataType::Binary => "binary".to_string(),
            DataType::Utf8 => "string".to_string(),
            DataType::LargeBinary => "largebinary".to_string(),
            DataType::LargeUtf8 => "largestring".to_string(),
            DataType::FixedSizeBinary(len) => format!("fixed_size_binary:{}", len),
            DataType::FixedSizeList(v, len) => {
                format!("fixed_size_list:{}:{}", v.data_type().type_str(), len)
            }
            x => format!("not supported format: {:?}", x),
        }
    }

    fn insert(&mut self, child: Field) {
        self.children.push(child)
    }

    fn field_mut(&mut self, id: i32) -> Option<&mut Field> {
        for field in &mut self.children {
            if field.id == id {
                return Some(field);
            }
            match field.field_mut(id) {
                Some(c) => return Some(c),
                None => {}
            }
        }
        None
    }
}

trait ToLogicalType {
    /// Return Arrow Data Type name.
    fn type_str(&self) -> String;

    fn is_numeric(&self) -> bool;
}

impl ToLogicalType for DataType {
    /// Return Arrow Data Type name.
    fn type_str(&self) -> String {
        match self {
            DataType::Boolean => "bool".to_string(),
            DataType::UInt8 => "uint8".to_string(),
            DataType::Int8 => "int8".to_string(),
            DataType::UInt16 => "uint16".to_string(),
            DataType::Int16 => "int16".to_string(),
            DataType::UInt32 => "uint32".to_string(),
            DataType::Int32 => "int32".to_string(),
            DataType::UInt64 => "uint64".to_string(),
            DataType::Int64 => "int64".to_string(),
            DataType::Float16 => "halffloat".to_string(),
            DataType::Float32 => "float".to_string(),
            DataType::Float64 => "double".to_string(),
            DataType::Date32 => "date32:day".to_string(),
            DataType::Date64 => "date64:ms".to_string(),
            DataType::Time32(unit) => format!("time32:{}", to_str(unit)),
            DataType::Time64(unit) => format!("time64:{}", to_str(unit)),
            DataType::Timestamp(unit, _) => format!("timestamp:{}", to_str(unit)),
            DataType::Binary => "binary".to_string(),
            DataType::Utf8 => "string".to_string(),
            DataType::LargeBinary => "largebinary".to_string(),
            DataType::LargeUtf8 => "largestring".to_string(),
            DataType::FixedSizeBinary(len) => format!("fixed_size_binary:{}", len),
            DataType::FixedSizeList(v, len) => {
                format!("fixed_size_list:{}:{}", Self::type_str(v.data_type()), len)
            }
            x => format!("not supported format: {:?}", x),
        }
    }

    fn is_numeric(&self) -> bool {
        use DataType::*;
        matches!(
            self,
            UInt8 | UInt16 | UInt32 | UInt64 | Int8 | Int16 | Int32 | Int64 | Float32 | Float64
        )
    }
}

fn to_str(unit: &TimeUnit) -> &'static str {
    match unit {
        TimeUnit::Second => "s",
        TimeUnit::Millisecond => "ms",
        TimeUnit::Microsecond => "us",
        TimeUnit::Nanosecond => "ns",
    }
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Field({}, id={}, type={}, ext_type={}, encoding={})",
            self.name,
            self.id,
            self.logical_type,
            self.extension_name,
            match &self.encoding {
                Some(enc) => format!("{}", enc),
                None => String::from("N/A"),
            }
        )?;
        self.children.iter().for_each(|field| {
            write!(f, "{}", field).unwrap();
        });
        Ok(())
    }
}

/// Lance file Schema.
#[derive(Debug, Clone)]
pub struct Schema {
    pub fields: Vec<Field>,
}

impl Schema {
    /// Create a Schema from arrow schema.
    pub fn new(schema: &arrow2::datatypes::Schema) -> Schema {
        Schema {
            fields: schema.fields.iter().map(Field::new).collect(),
        }
    }
    /// Create a new schema from protobuf.
    pub fn from_proto(fields: &[crate::format::pb::Field]) -> Schema {
        let mut schema = Schema { fields: vec![] };
        fields.iter().for_each(|f| {
            let lance_field = Field::from_proto(f);
            if lance_field.parent_id < 0 {
                schema.fields.push(lance_field);
            } else {
                schema
                    .field_mut(lance_field.parent_id)
                    .map(|f| f.insert(lance_field));
            }
        });
        schema
    }

    fn field_mut(&mut self, id: i32) -> Option<&mut Field> {
        for field in &mut self.fields {
            if field.id == id {
                return Some(field);
            }
            match field.field_mut(id) {
                Some(c) => return Some(c),
                None => {}
            }
        }
        None
    }
}

impl fmt::Display for Schema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Schema(")?;
        for field in &self.fields {
            write!(f, "{}", field)?
        }
        writeln!(f, ")")
    }
}
