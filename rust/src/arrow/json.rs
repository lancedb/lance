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

use crate::error::{Error, Result};

#[derive(Serialize, Deserialize, Debug)]
struct JsonDataType {
    #[serde(rename = "type")]
    type_: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    fields: Option<Vec<JsonField>>,
}

impl JsonDataType {
    fn try_new(dt: DataType) -> Result<Self> {
        dt.try_into()
    }

    fn new(type_name: &str) -> Self {
        Self {
            type_: type_name.to_string(),
            fields: None,
        }
    }
}

impl TryFrom<DataType> for JsonDataType {
    type Error = Error;

    fn try_from(dt: DataType) -> Result<Self> {
        let json_type = match dt {
            DataType::Null => Self::new("null"),
            DataType::Boolean => Self::new("boolean"),
            DataType::Int8 => Self::new("int8"),
            DataType::Int16 => Self::new("int16"),
            DataType::Int32 => Self::new("int32"),
            DataType::Int64 => Self::new("int64"),
            DataType::UInt8 => Self::new("uint8"),
            DataType::UInt16 => Self::new("uint16"),
            DataType::UInt32 => Self::new("uint32"),
            DataType::UInt64 => Self::new("uint64"),
            DataType::Float16 => todo!(),
            DataType::Float32 => todo!(),
            DataType::Float64 => todo!(),
            DataType::Timestamp(_, _) => todo!(),
            DataType::Date32 => todo!(),
            DataType::Date64 => todo!(),
            DataType::Time32(_) => todo!(),
            DataType::Time64(_) => todo!(),
            DataType::Duration(_) => todo!(),
            DataType::Interval(_) => todo!(),
            DataType::Binary => todo!(),
            DataType::FixedSizeBinary(_) => todo!(),
            DataType::LargeBinary => todo!(),
            DataType::Utf8 => Self::new("string"),
            DataType::LargeUtf8 => Self::new("large_string"),
            DataType::List(_) => todo!(),
            DataType::FixedSizeList(_, _) => todo!(),
            DataType::LargeList(_) => todo!(),
            DataType::Struct(_) => todo!(),
            DataType::Union(_, _) => todo!(),
            DataType::Dictionary(_, _) => todo!(),
            DataType::Decimal128(_, _) => todo!(),
            DataType::Decimal256(_, _) => todo!(),
            DataType::Map(_, _) => todo!(),
            DataType::RunEndEncoded(_, _) => todo!(),
        };
        Ok(json_type)
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct JsonField {
    name: String,
    nullable: bool,
    data_type: JsonDataType,
    children: Option<Vec<JsonField>>,
    metadata: Option<HashMap<String, String>>,
}

#[derive(Serialize, Deserialize, Debug)]
struct JsonSchema {
    fields: Vec<JsonField>,
    metadata: Option<HashMap<String, String>>,
}

pub trait ArrowJsonExt {
    fn to_json(&self) -> String;

    fn from_json(json: &str) -> Self;
}

impl ArrowJsonExt for Schema {
    fn to_json(&self) -> String {
        todo!()
    }

    fn from_json(json: &str) -> Self {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use serde_json;

    #[test]
    fn test_data_type_to_json() {
        assert_eq!(
            serde_json::to_string(&JsonDataType::try_new(DataType::Int32).unwrap()).unwrap(),
            r#"{"type":"int32"}"#
        );
        assert_eq!(
            serde_json::to_string(&JsonDataType::try_new(DataType::Boolean).unwrap()).unwrap(),
            r#"{"type":"boolean"}"#
        );
    }

    #[test]
    fn test_schema_to_json() {}
}
