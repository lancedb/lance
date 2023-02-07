// Copyright 2023 Lance Developers
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

//! Arrow / DuckDB conversion.

use std::collections::HashMap;

use arrow_array::{
    cast::as_primitive_array, types::*, Array, ArrowPrimitiveType, PrimitiveArray, RecordBatch,
};
use arrow_schema::DataType;
use duckdb_extension_framework::{duckly::idx_t, DataChunk, LogicalType, LogicalTypeId, Vector};

use crate::{Error, Result};

pub fn to_duckdb_type_id(data_type: &DataType) -> Result<LogicalTypeId> {
    use LogicalTypeId::*;

    let type_id = match data_type {
        DataType::Boolean => Boolean,
        DataType::Int8 => Tinyint,
        DataType::Int16 => Smallint,
        DataType::Int32 => Integer,
        DataType::Int64 => Bigint,
        DataType::UInt8 => Utinyint,
        DataType::UInt16 => Usmallint,
        DataType::UInt32 => Uinteger,
        DataType::UInt64 => Ubigint,
        DataType::Float32 => Float,
        DataType::Float64 => Double,
        DataType::Timestamp(_, _) => Timestamp,
        DataType::Date32 => Time,
        DataType::Date64 => Time,
        DataType::Time32(_) => Time,
        DataType::Time64(_) => Time,
        DataType::Duration(_) => todo!(),
        DataType::Interval(_) => Interval,
        DataType::Binary => Blob,
        DataType::FixedSizeBinary(_) => Blob,
        DataType::LargeBinary => Blob,
        DataType::Utf8 => Varchar,
        DataType::LargeUtf8 => Varchar,
        DataType::List(_) => List,
        DataType::FixedSizeList(_, _) => List,
        DataType::LargeList(_) => List,
        DataType::Struct(_) => Struct,
        DataType::Union(_, _, _) => Union,
        DataType::Dictionary(_, _) => todo!(),
        DataType::Decimal128(_, _) => Decimal,
        DataType::Decimal256(_, _) => Decimal,
        DataType::Map(_, _) => Map,
        _ => {
            return Err(Error::DuckDB(format!(
                "Unsupported arrow type: {data_type}"
            )));
        }
    };
    Ok(type_id)
}

pub fn to_duckdb_logical_type(data_type: &DataType) -> Result<LogicalType> {
    if data_type.is_primitive() {
        return Ok(LogicalType::new(to_duckdb_type_id(data_type)?));
    } else if let DataType::Struct(fields) = data_type {
        let mut shape = HashMap::new();
        for field in fields.iter() {
            shape.insert(
                field.name().as_str(),
                to_duckdb_logical_type(&field.data_type())?,
            );
        }
        return Ok(LogicalType::new_struct_type(shape));
    } else if let DataType::List(child) = data_type {
        return Ok(LogicalType::new_list_type(&to_duckdb_logical_type(
            child.data_type(),
        )?));
    } else if let DataType::LargeList(child) = data_type {
        return Ok(LogicalType::new_list_type(&to_duckdb_logical_type(
            child.data_type(),
        )?));
    } else if let DataType::FixedSizeList(child, _) = data_type {
        return Ok(LogicalType::new_list_type(&to_duckdb_logical_type(
            child.data_type(),
        )?));
    }
    todo!()
}

pub fn record_batch_to_duckdb_data_chunk(batch: &RecordBatch, chunk: &mut DataChunk) -> Result<()> {
    // Fill the row
    for i in 0..batch.num_columns() {
        let col = batch.column(i);
        match col.data_type() {
            DataType::UInt8 => {
                primitive_array_to_duckdb_vector::<UInt8Type>(
                    as_primitive_array(col.as_ref()),
                    &mut chunk.get_vector(i as idx_t),
                );
            }
            DataType::UInt16 => {
                primitive_array_to_duckdb_vector::<UInt16Type>(
                    as_primitive_array(col.as_ref()),
                    &mut chunk.get_vector(i as idx_t),
                );
            }
            DataType::UInt32 => {
                primitive_array_to_duckdb_vector::<UInt32Type>(
                    as_primitive_array(col.as_ref()),
                    &mut chunk.get_vector(i as idx_t),
                );
            }
            DataType::UInt64 => {
                primitive_array_to_duckdb_vector::<UInt64Type>(
                    as_primitive_array(col.as_ref()),
                    &mut chunk.get_vector(i as idx_t),
                );
            }
            DataType::Int8 => {
                primitive_array_to_duckdb_vector::<Int8Type>(
                    as_primitive_array(col.as_ref()),
                    &mut chunk.get_vector(i as idx_t),
                );
            }
            DataType::Int16 => {
                primitive_array_to_duckdb_vector::<Int16Type>(
                    as_primitive_array(col.as_ref()),
                    &mut chunk.get_vector(i as idx_t),
                );
            }
            DataType::Int32 => {
                primitive_array_to_duckdb_vector::<Int32Type>(
                    as_primitive_array(col.as_ref()),
                    &mut chunk.get_vector(i as idx_t),
                );
            }
            DataType::Int64 => {
                primitive_array_to_duckdb_vector::<Int64Type>(
                    as_primitive_array(col.as_ref()),
                    &mut chunk.get_vector(i as idx_t),
                );
            }
            DataType::Float32 => {
                primitive_array_to_duckdb_vector::<Float32Type>(
                    as_primitive_array(col.as_ref()),
                    &mut chunk.get_vector(i as idx_t),
                );
            }
            DataType::Float64 => {
                primitive_array_to_duckdb_vector::<Float64Type>(
                    as_primitive_array(col.as_ref()),
                    &mut chunk.get_vector(i as idx_t),
                );
            },
            DataType::Utf8 => {

            },
            _ => {
                println!("column {} is not supported yet, please file an issue https://github.com/eto-ai/lance", batch.schema().field(i));
            }
        }
    }
    chunk.set_size(batch.num_columns() as idx_t);
    Ok(())
}

pub fn primitive_array_to_duckdb_vector<T: ArrowPrimitiveType>(
    array: &PrimitiveArray<T>,
    out_vector: &mut Vector<T::Native>,
) {
    println!(
        "Array len: {} out vector len: {}",
        array.len(),
        out_vector.get_data_as_slice().len()
    );
    assert!(array.len() <= out_vector.get_data_as_slice().len(),);

    for i in 0..array.len() {
        out_vector.get_data_as_slice()[i] = array.value(i);
    }
}
