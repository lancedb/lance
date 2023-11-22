use crate::arrow_array::{
    BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
    StringArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use crate::arrow_schema::DataType;

pub trait ArrowTypeInfer {
    type ArrayType;

    fn arrow_type() -> DataType;
}

impl ArrowTypeInfer for bool {
    type ArrayType = BooleanArray;

    fn arrow_type() -> DataType {
        DataType::Boolean
    }
}

impl ArrowTypeInfer for i8 {
    type ArrayType = Int8Array;

    fn arrow_type() -> DataType {
        DataType::Int8
    }
}

impl ArrowTypeInfer for i16 {
    type ArrayType = Int16Array;

    fn arrow_type() -> DataType {
        DataType::Int16
    }
}

impl ArrowTypeInfer for i32 {
    type ArrayType = Int32Array;

    fn arrow_type() -> DataType {
        DataType::Int32
    }
}

impl ArrowTypeInfer for i64 {
    type ArrayType = Int64Array;

    fn arrow_type() -> DataType {
        DataType::Int64
    }
}

impl ArrowTypeInfer for u8 {
    type ArrayType = UInt8Array;

    fn arrow_type() -> DataType {
        DataType::UInt8
    }
}

impl ArrowTypeInfer for u16 {
    type ArrayType = UInt16Array;

    fn arrow_type() -> DataType {
        DataType::UInt16
    }
}

impl ArrowTypeInfer for u32 {
    type ArrayType = UInt32Array;

    fn arrow_type() -> DataType {
        DataType::UInt32
    }
}

impl ArrowTypeInfer for u64 {
    type ArrayType = UInt64Array;

    fn arrow_type() -> DataType {
        DataType::UInt64
    }
}

impl ArrowTypeInfer for f32 {
    type ArrayType = Float32Array;

    fn arrow_type() -> DataType {
        DataType::Float32
    }
}

impl ArrowTypeInfer for f64 {
    type ArrayType = Float64Array;

    fn arrow_type() -> DataType {
        DataType::Float64
    }
}

impl ArrowTypeInfer for &str {
    type ArrayType = StringArray;

    fn arrow_type() -> DataType {
        DataType::Utf8
    }
}

impl ArrowTypeInfer for String {
    type ArrayType = StringArray;

    fn arrow_type() -> DataType {
        DataType::Utf8
    }
}
