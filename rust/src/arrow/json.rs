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

//! Serialization and deserialization of Arrow Schema to JSON.

use std::collections::HashMap;

use arrow_schema::{DataType, Field, Schema};
use serde::{Deserialize, Serialize};

use crate::datatypes::LogicalType;
use crate::error::{Error, Result};

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Self::Arrow {
            message: e.to_string(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct JsonDataType {
    #[serde(rename = "type")]
    type_: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    fields: Option<Vec<JsonField>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    length: Option<usize>,
}

impl JsonDataType {
    fn try_new(dt: &DataType) -> Result<Self> {
        dt.try_into()
    }
}

impl TryFrom<&DataType> for JsonDataType {
    type Error = Error;

    fn try_from(dt: &DataType) -> Result<Self> {
        let (type_name, fields) = match dt {
            DataType::Null
            | DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Utf8
            | DataType::Binary
            | DataType::LargeUtf8
            | DataType::LargeBinary
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::Duration(_)
            | DataType::Interval(_)
            | DataType::Dictionary(_, _) => {
                let logical_type: LogicalType = dt.try_into()?;
                (logical_type.to_string(), None)
            }

            DataType::List(f) => {
                let fields = vec![JsonField::try_from(f.as_ref())?];
                ("list".to_string(), Some(fields))
            }
            DataType::LargeList(f) => {
                let fields = vec![JsonField::try_from(f.as_ref())?];
                ("large_list".to_string(), Some(fields))
            }
            DataType::FixedSizeList(f, len) => {
                let fields = vec![JsonField::try_from(f.as_ref())?];
                return Ok(Self {
                    type_: "fixed_size_list".to_string(),
                    fields: Some(fields),
                    length: Some(*len as usize),
                });
            }
            DataType::Struct(fields) => {
                let fields = fields
                    .iter()
                    .map(|f| JsonField::try_from(f.as_ref()))
                    .collect::<Result<Vec<_>>>()?;
                ("struct".to_string(), Some(fields))
            }
            _ => {
                return Err(Error::Arrow {
                    message: format!("Json conversion: Unsupported type: {dt}"),
                })
            }
        };

        Ok(Self {
            type_: type_name,
            fields,
            length: None,
        })
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct JsonField {
    name: String,
    #[serde(rename = "type")]
    type_: JsonDataType,
    nullable: bool,
}

impl TryFrom<&Field> for JsonField {
    type Error = Error;

    fn try_from(field: &Field) -> Result<Self> {
        let data_type = JsonDataType::try_new(field.data_type())?;

        Ok(Self {
            name: field.name().to_string(),
            nullable: field.is_nullable(),
            type_: data_type,
        })
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct JsonSchema {
    fields: Vec<JsonField>,

    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<HashMap<String, String>>,
}

/// Convert Schema to JSON representation.
impl TryFrom<&Schema> for JsonSchema {
    type Error = Error;

    fn try_from(schema: &Schema) -> Result<Self> {
        let fields = schema
            .fields()
            .iter()
            .map(|f| JsonField::try_from(f.as_ref()))
            .collect::<Result<Vec<_>>>()?;

        let metadata = if schema.metadata.is_empty() {
            None
        } else {
            Some(schema.metadata.clone())
        };
        Ok(Self { fields, metadata })
    }
}

pub trait ArrowJsonExt {
    fn to_json(&self) -> Result<String>;

    fn from_json(json: &str) -> Self;
}

impl ArrowJsonExt for Schema {
    fn to_json(&self) -> Result<String> {
        let json_schema = JsonSchema::try_from(self)?;
        Ok(serde_json::to_string(&json_schema)?)
    }

    fn from_json(json: &str) -> Self {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use std::sync::Arc;

    use arrow_schema::TimeUnit;
    use serde_json;
    use serde_json::{json, Value};

    fn assert_type_json_str(dt: DataType, val: Value) {
        assert_eq!(
            serde_json::from_str::<Value>(
                &serde_json::to_string(&JsonDataType::try_new(&dt).unwrap()).unwrap()
            )
            .unwrap(),
            val
        );
    }

    fn assert_primitive_types(dt: DataType, type_name: &str) {
        assert_type_json_str(dt, json!({"type": type_name}));
    }

    #[test]
    fn test_data_type_to_json() {
        assert_primitive_types(DataType::Null, "null");
        assert_primitive_types(DataType::Boolean, "bool");
        assert_primitive_types(DataType::Int8, "int8");
        assert_primitive_types(DataType::Int16, "int16");
        assert_primitive_types(DataType::Int32, "int32");
        assert_primitive_types(DataType::Int64, "int64");
        assert_primitive_types(DataType::UInt8, "uint8");
        assert_primitive_types(DataType::UInt16, "uint16");
        assert_primitive_types(DataType::UInt32, "uint32");
        assert_primitive_types(DataType::UInt64, "uint64");
        assert_primitive_types(DataType::Float16, "halffloat");
        assert_primitive_types(DataType::Float32, "float");
        assert_primitive_types(DataType::Float64, "double");
        assert_primitive_types(DataType::Utf8, "string");
        assert_primitive_types(DataType::LargeUtf8, "large_string");
        assert_primitive_types(DataType::Binary, "binary");
        assert_primitive_types(DataType::LargeBinary, "large_binary");
        assert_primitive_types(DataType::Date32, "date32:day");
        assert_primitive_types(DataType::Date64, "date64:ms");
        assert_primitive_types(DataType::Time32(TimeUnit::Second), "time32:s");
    }

    #[test]
    fn test_complex_types_to_json() {
        assert_type_json_str(
            DataType::List(Arc::new(Field::new("item", DataType::Float32, false))),
            json!(
                {
                    "type": "list",
                    "fields": [
                        {
                            "name": "item",
                            "type": {
                                "type": "float"
                            },
                            "nullable": false
                        }
                    ]
                }
            ),
        );

        assert_type_json_str(
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 32),
            json!(
                {
                    "type": "fixed_size_list",
                    "fields": [
                        {
                            "name": "item",
                            "type": {
                                "type": "float"
                            },
                            "nullable": false
                        }
                    ],
                    "length": 32
                }
            ),
        );

        assert_type_json_str(
            DataType::Struct(
                vec![
                    Field::new("a", DataType::Date32, false),
                    Field::new("b", DataType::Int32, true),
                ]
                .into(),
            ),
            json!({
                "type": "struct",
                "fields": [
                    {
                        "name": "a",
                        "type": {
                            "type": "date32:day"
                        },
                        "nullable": false
                    },
                    {
                        "name": "b",
                        "type": {
                            "type": "int32"
                        },
                        "nullable": true
                    }
                ]
            }),
        );
    }

    #[test]
    fn test_schema_to_json() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Date32, false),
            Field::new("b", DataType::Int32, true),
            Field::new(
                "s",
                DataType::Struct(vec![Field::new("str", DataType::Utf8, false)].into()),
                false,
            ),
        ]);
        assert_eq!(
            serde_json::from_str::<Value>(&schema.to_json().unwrap()).unwrap(),
            json!({
                "fields": [
                    {
                        "name": "a",
                        "type": {
                            "type": "date32:day"
                        },
                        "nullable": false
                    },
                    {
                        "name": "b",
                        "type": {
                            "type": "int32"
                        },
                        "nullable": true
                    },
                    {
                        "name": "s",
                        "type": {
                            "type": "struct",
                            "fields": [
                                {
                                    "name": "str",
                                    "type": {
                                        "type": "string"
                                    },
                                    "nullable": false
                                }
                            ]
                        },
                        "nullable": false
                    },
                ]
            })
        );
    }
}
