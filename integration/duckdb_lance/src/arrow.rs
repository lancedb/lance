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

use arrow_array::{
    cast::{
        as_boolean_array, as_large_list_array, as_list_array, as_primitive_array, as_string_array,
        as_struct_array,
    },
    types::*,
    Array, ArrowPrimitiveType, BooleanArray, FixedSizeListArray, GenericListArray, OffsetSizeTrait,
    PrimitiveArray, RecordBatch, StringArray, StructArray,
};
use arrow_schema::DataType;
use duckdb_ext::{DataChunk, FlatVector, Inserter, ListVector, StructVector, Vector};
use duckdb_ext::{LogicalType, LogicalTypeId};
use lance::arrow::as_fixed_size_list_array;
use num_traits::AsPrimitive;

use crate::{Error, Result};

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
        DataType::Duration(_) => Interval,
        DataType::Interval(_) => Interval,
        DataType::Binary | DataType::LargeBinary | DataType::FixedSizeBinary(_) => Blob,
        DataType::Utf8 | DataType::LargeUtf8 => Varchar,
        DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _) => List,
        DataType::Struct(_) => Struct,
        DataType::Union(_, _) => Union,
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
        todo!("Unsupported data type: {data_type}, please file an issue at https://github.com/lancedb/lance");
    }
}

pub fn record_batch_to_duckdb_data_chunk(batch: &RecordBatch, chunk: &mut DataChunk) -> Result<()> {
    // Fill the row
    assert_eq!(batch.num_columns(), chunk.num_columns());
    for i in 0..batch.num_columns() {
        let col = batch.column(i);
        match col.data_type() {
            dt if dt.is_primitive() || matches!(dt, DataType::Boolean) => {
                primitive_array_to_vector(col, &mut chunk.flat_vector(i));
            }
            DataType::Utf8 => {
                string_array_to_vector(as_string_array(col.as_ref()), &mut chunk.flat_vector(i));
            }
            DataType::List(_) => {
                list_array_to_vector(as_list_array(col.as_ref()), &mut chunk.list_vector(i));
            }
            DataType::LargeList(_) => {
                list_array_to_vector(as_large_list_array(col.as_ref()), &mut chunk.list_vector(i));
            }
            DataType::FixedSizeList(_, _) => {
                fixed_size_list_array_to_vector(
                    as_fixed_size_list_array(col.as_ref()),
                    &mut chunk.list_vector(i),
                );
            }
            DataType::Struct(_) => {
                let struct_array = as_struct_array(col.as_ref());
                let mut struct_vector = chunk.struct_vector(i);
                struct_array_to_vector(struct_array, &mut struct_vector);
            }
            _ => {
                todo!("column {} is not supported yet, please file an issue at https://github.com/lancedb/lance", batch.schema().field(i));
            }
        }
    }
    chunk.set_len(batch.num_rows());
    Ok(())
}

fn primitive_array_to_flat_vector<T: ArrowPrimitiveType>(
    array: &PrimitiveArray<T>,
    out_vector: &mut FlatVector,
) {
    // assert!(array.len() <= out_vector.capacity());
    out_vector.copy::<T::Native>(array.values());
}

fn primitive_array_to_vector(array: &dyn Array, out: &mut dyn Vector) {
    match array.data_type() {
        DataType::Boolean => {
            boolean_array_to_vector(
                as_boolean_array(array),
                out.as_mut_any().downcast_mut().unwrap(),
            );
        }
        DataType::UInt8 => {
            primitive_array_to_flat_vector::<UInt8Type>(
                as_primitive_array(array),
                out.as_mut_any().downcast_mut().unwrap(),
            );
        }
        DataType::UInt16 => {
            primitive_array_to_flat_vector::<UInt16Type>(
                as_primitive_array(array),
                out.as_mut_any().downcast_mut().unwrap(),
            );
        }
        DataType::UInt32 => {
            primitive_array_to_flat_vector::<UInt32Type>(
                as_primitive_array(array),
                out.as_mut_any().downcast_mut().unwrap(),
            );
        }
        DataType::UInt64 => {
            primitive_array_to_flat_vector::<UInt64Type>(
                as_primitive_array(array),
                out.as_mut_any().downcast_mut().unwrap(),
            );
        }
        DataType::Int8 => {
            primitive_array_to_flat_vector::<Int8Type>(
                as_primitive_array(array),
                out.as_mut_any().downcast_mut().unwrap(),
            );
        }
        DataType::Int16 => {
            primitive_array_to_flat_vector::<Int16Type>(
                as_primitive_array(array),
                out.as_mut_any().downcast_mut().unwrap(),
            );
        }
        DataType::Int32 => {
            primitive_array_to_flat_vector::<Int32Type>(
                as_primitive_array(array),
                out.as_mut_any().downcast_mut().unwrap(),
            );
        }
        DataType::Int64 => {
            primitive_array_to_flat_vector::<Int64Type>(
                as_primitive_array(array),
                out.as_mut_any().downcast_mut().unwrap(),
            );
        }
        DataType::Float32 => {
            primitive_array_to_flat_vector::<Float32Type>(
                as_primitive_array(array),
                out.as_mut_any().downcast_mut().unwrap(),
            );
        }
        DataType::Float64 => {
            primitive_array_to_flat_vector::<Float64Type>(
                as_primitive_array(array),
                out.as_mut_any().downcast_mut().unwrap(),
            );
        }
        _ => {
            todo!()
        }
    }
}

/// Convert Arrow [BooleanArray] to a duckdb vector.
fn boolean_array_to_vector(array: &BooleanArray, out: &mut FlatVector) {
    assert!(array.len() <= out.capacity());

    for i in 0..array.len() {
        out.as_mut_slice()[i] = array.value(i);
    }
}

fn string_array_to_vector(array: &StringArray, out: &mut FlatVector) {
    assert!(array.len() <= out.capacity());

    // TODO: zero copy assignment
    for i in 0..array.len() {
        let s = array.value(i);
        out.insert(i, s);
    }
}

fn list_array_to_vector<O: OffsetSizeTrait + AsPrimitive<usize>>(
    array: &GenericListArray<O>,
    out: &mut ListVector,
) {
    let value_array = array.values();
    let mut child = out.child(value_array.len());
    match value_array.data_type() {
        dt if dt.is_primitive() => {
            primitive_array_to_vector(value_array.as_ref(), &mut child);
            for i in 0..array.len() {
                let offset = array.value_offsets()[i];
                let length = array.value_length(i);
                out.set_entry(i, offset.as_(), length.as_());
            }
        }
        _ => {
            todo!("Nested list is not supported yet.");
        }
    }
}

fn fixed_size_list_array_to_vector(array: &FixedSizeListArray, out: &mut ListVector) {
    let value_array = array.values();
    let mut child = out.child(value_array.len());
    match value_array.data_type() {
        dt if dt.is_primitive() => {
            primitive_array_to_vector(value_array.as_ref(), &mut child);
            for i in 0..array.len() {
                let offset = array.value_offset(i);
                let length = array.value_length();
                out.set_entry(i, offset as usize, length as usize);
            }
            out.set_len(value_array.len());
        }
        _ => {
            todo!("Nested list is not supported yet.");
        }
    }
}

fn struct_array_to_vector(array: &StructArray, out: &mut StructVector) {
    for i in 0..array.num_columns() {
        let column = array.column(i);
        match column.data_type() {
            dt if dt.is_primitive() || matches!(dt, DataType::Boolean) => {
                primitive_array_to_vector(column, &mut out.child(i));
            }
            DataType::Utf8 => {
                string_array_to_vector(as_string_array(column.as_ref()), &mut out.child(i));
            }
            DataType::List(_) => {
                list_array_to_vector(
                    as_list_array(column.as_ref()),
                    &mut out.list_vector_child(i),
                );
            }
            DataType::LargeList(_) => {
                list_array_to_vector(
                    as_large_list_array(column.as_ref()),
                    &mut out.list_vector_child(i),
                );
            }
            DataType::FixedSizeList(_, _) => {
                fixed_size_list_array_to_vector(
                    as_fixed_size_list_array(column.as_ref()),
                    &mut out.list_vector_child(i),
                );
            }
            DataType::Struct(_) => {
                let struct_array = as_struct_array(column.as_ref());
                let mut struct_vector = out.struct_vector_child(i);
                struct_array_to_vector(struct_array, &mut struct_vector);
            }
            _ => {
                todo!("Unsupported data type: {}, please file an issue at https://github.com/lancedb/lance", column.data_type());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_schema::{Field, Schema};

    // use libduckdb to link to a duckdb binary.
    #[allow(unused_imports)]
    use libduckdb_sys;

    #[test]
    fn test_record_batch_to_data_chunk() {
        let schema = Arc::new(Schema::new(vec![Field::new("b", DataType::Boolean, false)]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(BooleanArray::from(vec![true, false, true]))],
        )
        .unwrap();

        let logical_types = schema
            .fields
            .iter()
            .map(|f| to_duckdb_logical_type(f.data_type()).unwrap())
            .collect::<Vec<_>>();
        let mut chunk = DataChunk::new(&logical_types);

        record_batch_to_duckdb_data_chunk(&batch, &mut chunk).unwrap();
        assert_eq!(chunk.len(), 3);
        let vector = chunk.flat_vector(0);
        assert_eq!(LogicalTypeId::Boolean, vector.logical_type().id());
        assert_eq!(vector.as_slice::<bool>()[0], true);
        assert_eq!(vector.as_slice::<bool>()[1], false);
        assert_eq!(vector.as_slice::<bool>()[2], true);
    }
}
