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

use crate::{Error, Result};
use arrow_array::{
    cast::{as_boolean_array, as_primitive_array, as_string_array, as_struct_array},
    types::*,
    Array, ArrowPrimitiveType, BooleanArray, PrimitiveArray, RecordBatch, StringArray, StructArray,
};
use arrow_schema::DataType;
use duckdb_ext::{DataChunk, Inserter, StructVector, Vector};
use duckdb_ext::{LogicalType, LogicalTypeId};

pub fn to_duckdb_type_id(data_type: &DataType) -> Result<LogicalTypeId> {
    use LogicalTypeId::*;

    let type_id = match data_type {
        DataType::Boolean => Boolean,
        DataType::Int8 => Tinyint,
        DataType::Int16 => Smallint,
        DataType::Int32 => Integer,
        DataType::Int64 => Bigint,
        DataType::UInt8 => UTinyint,
        DataType::UInt16 => USmallint,
        DataType::UInt32 => UInteger,
        DataType::UInt64 => UBigint,
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
    if data_type.is_primitive()
        || matches!(
            data_type,
            DataType::Boolean
                | DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Binary
                | DataType::LargeBinary
        )
    {
        Ok(LogicalType::new(to_duckdb_type_id(data_type)?))
    } else if let DataType::Dictionary(_, value_type) = data_type {
        to_duckdb_logical_type(value_type)
    } else if let DataType::Struct(fields) = data_type {
        let mut shape = vec![];
        for field in fields.iter() {
            shape.push((
                field.name().as_str(),
                to_duckdb_logical_type(field.data_type())?,
            ));
        }
        Ok(LogicalType::struct_type(shape.as_slice()))
    } else if let DataType::List(child) = data_type {
        Ok(LogicalType::list_type(&to_duckdb_logical_type(
            child.data_type(),
        )?))
    } else if let DataType::LargeList(child) = data_type {
        Ok(LogicalType::list_type(&to_duckdb_logical_type(
            child.data_type(),
        )?))
    } else if let DataType::FixedSizeList(child, _) = data_type {
        Ok(LogicalType::list_type(&to_duckdb_logical_type(
            child.data_type(),
        )?))
    } else {
        println!("Unsupported data type: {data_type}, please file an issue https://github.com/eto-ai/lance");
        todo!()
    }
}

pub fn record_batch_to_duckdb_data_chunk(batch: &RecordBatch, chunk: &mut DataChunk) -> Result<()> {
    // Fill the row
    for i in 0..batch.num_columns() {
        let col = batch.column(i);
        match col.data_type() {
            DataType::Boolean => {
                boolean_array_to_vector(as_boolean_array(col.as_ref()), &mut chunk.vector(i));
            }
            DataType::UInt8 => {
                primitive_array_to_duckdb_vector::<UInt8Type>(
                    as_primitive_array(col.as_ref()),
                    &mut chunk.vector(i),
                );
            }
            DataType::UInt16 => {
                primitive_array_to_duckdb_vector::<UInt16Type>(
                    as_primitive_array(col.as_ref()),
                    &mut chunk.vector(i),
                );
            }
            DataType::UInt32 => {
                primitive_array_to_duckdb_vector::<UInt32Type>(
                    as_primitive_array(col.as_ref()),
                    &mut chunk.vector(i),
                );
            }
            DataType::UInt64 => {
                primitive_array_to_duckdb_vector::<UInt64Type>(
                    as_primitive_array(col.as_ref()),
                    &mut chunk.vector(i),
                );
            }
            DataType::Int8 => {
                primitive_array_to_duckdb_vector::<Int8Type>(
                    as_primitive_array(col.as_ref()),
                    &mut chunk.vector(i),
                );
            }
            DataType::Int16 => {
                primitive_array_to_duckdb_vector::<Int16Type>(
                    as_primitive_array(col.as_ref()),
                    &mut chunk.vector(i),
                );
            }
            DataType::Int32 => {
                primitive_array_to_duckdb_vector::<Int32Type>(
                    as_primitive_array(col.as_ref()),
                    &mut chunk.vector(i),
                );
            }
            DataType::Int64 => {
                primitive_array_to_duckdb_vector::<Int64Type>(
                    as_primitive_array(col.as_ref()),
                    &mut chunk.vector(i),
                );
            }
            DataType::Float32 => {
                primitive_array_to_duckdb_vector::<Float32Type>(
                    as_primitive_array(col.as_ref()),
                    &mut chunk.vector(i),
                );
            }
            DataType::Float64 => {
                primitive_array_to_duckdb_vector::<Float64Type>(
                    as_primitive_array(col.as_ref()),
                    &mut chunk.vector(i),
                );
            }
            DataType::Utf8 => {
                string_array_to_vector(as_string_array(col.as_ref()), &mut chunk.vector::<&str>(i));
            }
            DataType::Struct(_) => {
                let struct_array = as_struct_array(col.as_ref());
                let mut struct_vector = chunk.struct_vector(i);
                struct_array_to_vector(struct_array, &mut struct_vector);
            }
            _ => {
                println!("column {} is not supported yet, please file an issue https://github.com/eto-ai/lance", batch.schema().field(i));
            }
        }
    }
    chunk.set_len(batch.num_rows());
    Ok(())
}

fn primitive_array_to_duckdb_vector<T: ArrowPrimitiveType>(
    array: &PrimitiveArray<T>,
    out_vector: &mut Vector<T::Native>,
) {
    assert!(array.len() <= out_vector.capacity());
    out_vector.copy(array.values());
}

/// Convert Arrow [BooleanArray] to a duckdb vector.
fn boolean_array_to_vector(array: &BooleanArray, out: &mut Vector<bool>) {
    assert!(array.len() <= out.capacity());

    for i in 0..array.len() {
        out.as_mut_slice()[i] = array.value(i);
    }
}

fn string_array_to_vector(array: &StringArray, out: &mut Vector<&str>) {
    assert!(array.len() <= out.capacity());

    // TODO: zero copy assignment
    for i in 0..array.len() {
        let s = array.value(i);
        out.insert(i, s);
    }
}

fn struct_array_to_vector(array: &StructArray, out: &mut StructVector) {
    for i in 0..array.num_columns() {
        let column = array.column(i);
        match column.data_type() {
            DataType::Boolean => {
                boolean_array_to_vector(as_boolean_array(column.as_ref()), &mut out.child(i));
            }
            DataType::UInt8 => {
                primitive_array_to_duckdb_vector::<UInt8Type>(
                    as_primitive_array(column.as_ref()),
                    &mut out.child(i),
                );
            }
            DataType::UInt16 => {
                primitive_array_to_duckdb_vector::<UInt16Type>(
                    as_primitive_array(column.as_ref()),
                    &mut out.child(i),
                );
            }
            DataType::UInt32 => {
                primitive_array_to_duckdb_vector::<UInt32Type>(
                    as_primitive_array(column.as_ref()),
                    &mut out.child(i),
                );
            }
            DataType::UInt64 => {
                primitive_array_to_duckdb_vector::<UInt64Type>(
                    as_primitive_array(column.as_ref()),
                    &mut out.child(i),
                );
            }
            DataType::Int8 => {
                primitive_array_to_duckdb_vector::<Int8Type>(
                    as_primitive_array(column.as_ref()),
                    &mut out.child(i),
                );
            }
            DataType::Int16 => {
                primitive_array_to_duckdb_vector::<Int16Type>(
                    as_primitive_array(column.as_ref()),
                    &mut out.child(i),
                );
            }
            DataType::Int32 => {
                primitive_array_to_duckdb_vector::<Int32Type>(
                    as_primitive_array(column.as_ref()),
                    &mut out.child(i),
                );
            }
            DataType::Int64 => {
                primitive_array_to_duckdb_vector::<Int64Type>(
                    as_primitive_array(column.as_ref()),
                    &mut out.child(i),
                );
            }
            DataType::Float32 => {
                primitive_array_to_duckdb_vector::<Float32Type>(
                    as_primitive_array(column.as_ref()),
                    &mut out.child(i),
                );
            }
            DataType::Float64 => {
                primitive_array_to_duckdb_vector::<Float64Type>(
                    as_primitive_array(column.as_ref()),
                    &mut out.child(i),
                );
            }
            DataType::Utf8 => {
                string_array_to_vector(as_string_array(column.as_ref()), &mut out.child(i));
            }
            DataType::Struct(_) => {
                let struct_array = as_struct_array(column.as_ref());
                let mut struct_vector = out.struct_vector_child(i);
                struct_array_to_vector(struct_array, &mut struct_vector);
            }
            _ => {
                println!("Unsupported data type: {}, please file an issue https://github.com/eto-ai/lance", column.data_type());
                todo!()
            }
        }
    }
}
