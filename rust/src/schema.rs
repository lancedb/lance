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

use crate::format::pb;
use arrow::datatypes::DataType;
use std::fmt;

use crate::encodings::Encoding;

/// Lance Field.
///
/// Metadata of a column.
#[derive(Debug)]
pub struct Field {
    pub id: i32,
    pub parent_id: i32,
    pub name: String,
    logical_type: String,
    extension_name: String,
    pub encoding: Option<Encoding>,
    node_type: i32,

    children: Vec<Field>,
}

impl Field {
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
    pub fn datatype(&self) -> DataType {
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
            _ => panic!(),
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
#[derive(Debug)]
pub struct Schema {
    fields: Vec<Field>,
}

impl Schema {
    /// Create a new schema from protobuf.
    pub fn from_proto(fields: &Vec<crate::format::pb::Field>) -> Schema {
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
