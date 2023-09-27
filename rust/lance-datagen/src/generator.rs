use std::{iter, marker::PhantomData, sync::Arc};

use arrow::{
    array::ArrayData,
    buffer::{BooleanBuffer, Buffer, OffsetBuffer},
};
use arrow_array::{
    make_array,
    types::{ArrowDictionaryKeyType, BinaryType, ByteArrayType, Utf8Type},
    Array, FixedSizeListArray, RecordBatch, RecordBatchReader,
};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use rand::{Rng, RngCore, SeedableRng};

#[derive(Copy, Clone, Debug, Default)]
pub struct RowCount(u64);
#[derive(Copy, Clone, Debug, Default)]
pub struct BatchCount(u32);
#[derive(Copy, Clone, Debug, Default)]
pub struct ByteCount(u64);
#[derive(Copy, Clone, Debug, Default)]
pub struct Dimension(u32);

impl From<u32> for BatchCount {
    fn from(n: u32) -> Self {
        Self(n)
    }
}

impl From<u64> for RowCount {
    fn from(n: u64) -> Self {
        Self(n)
    }
}

impl From<u64> for ByteCount {
    fn from(n: u64) -> Self {
        Self(n)
    }
}

impl From<u32> for Dimension {
    fn from(n: u32) -> Self {
        Self(n)
    }
}

/// A trait for anything that can generate arrays of data
pub trait ArrayGenerator {
    /// Generate an array of the given length
    ///
    /// # Arguments
    ///
    /// * `length` - The number of elements to generate
    /// * `rng` - The random number generator to use
    ///
    /// # Returns
    ///
    /// An array of the given length
    ///
    /// Note: Not every generator needs an rng.  However, it is passed here because many do and this
    /// lets us manage RNGs at the batch level instead of the array level.
    fn generate(
        &mut self,
        length: RowCount,
        rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
    ) -> Result<Arc<dyn arrow_array::Array>, ArrowError>;
    /// Get the data type of the array that this generator produces
    ///
    /// # Returns
    ///
    /// The data type of the array that this generator produces
    fn data_type(&self) -> &DataType;
    /// Get the size of each element in bytes
    ///
    /// # Returns
    ///
    /// The size of each element in bytes.  Will be None if the size varies by element.
    fn element_size_bytes(&self) -> Option<ByteCount>;
}

pub struct NullGenerator {
    generator: Box<dyn ArrayGenerator>,
    null_probability: f64,
}

impl ArrayGenerator for NullGenerator {
    fn generate(
        &mut self,
        length: RowCount,
        rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
    ) -> Result<Arc<dyn arrow_array::Array>, ArrowError> {
        let array = self.generator.generate(length, rng)?;
        let data = array.to_data();

        if self.null_probability < 0.0 || self.null_probability > 1.0 {
            return Err(ArrowError::InvalidArgumentError(format!(
                "null_probability must be between 0 and 1, got {}",
                self.null_probability
            )));
        }

        let (null_count, new_validity) = if self.null_probability == 0.0 {
            if data.null_count() == 0 {
                return Ok(array);
            } else {
                (0_usize, None)
            }
        } else if self.null_probability == 1.0 {
            if data.null_count() == data.len() {
                return Ok(array);
            } else {
                let all_nulls = BooleanBuffer::new_unset(array.len());
                (array.len(), Some(all_nulls.into_inner()))
            }
        } else {
            let array_len = array.len();
            let num_validity_bytes = (array_len + 7) / 8;
            let mut null_count = 0;
            // Sampling the RNG once per bit is kind of slow so we do this to sample once
            // per byte.  We only get 8 bits of RNG resolution but that should be good enough.
            let threshold = (self.null_probability * std::u8::MAX as f64) as u8;
            let bytes = (0..num_validity_bytes)
                .map(|byte_idx| {
                    let mut sample = rng.gen::<u64>();
                    let mut byte: u8 = 0;
                    for bit_idx in 0..8 {
                        // We could probably overshoot and fill in extra bits with random data but
                        // this is cleaner and that would mess up the null count
                        byte <<= 1;
                        let pos = byte_idx * 8 + (7 - bit_idx);
                        if pos < array_len {
                            let sample_piece = sample & 0xFF;
                            let is_null = (sample_piece as u8) < threshold;
                            byte |= (!is_null) as u8;
                            null_count += is_null as usize;
                        }
                        sample >>= 8;
                    }
                    byte
                })
                .collect::<Vec<_>>();
            let new_validity = Buffer::from_iter(bytes);
            (null_count, Some(new_validity))
        };

        unsafe {
            let new_data = ArrayData::new_unchecked(
                data.data_type().clone(),
                data.len(),
                Some(null_count),
                new_validity,
                data.offset(),
                data.buffers().to_vec(),
                data.child_data().into(),
            );
            Ok(make_array(new_data))
        }
    }

    fn data_type(&self) -> &DataType {
        self.generator.data_type()
    }

    fn element_size_bytes(&self) -> Option<ByteCount> {
        self.generator.element_size_bytes()
    }
}

pub trait ArrayGeneratorExt {
    fn with_nulls(self, null_probability: f64) -> Box<dyn ArrayGenerator>;
}

impl ArrayGeneratorExt for Box<dyn ArrayGenerator> {
    fn with_nulls(self, null_probability: f64) -> Box<dyn ArrayGenerator> {
        Box::new(NullGenerator {
            generator: self,
            null_probability,
        })
    }
}

pub struct NTimesIter<I: Iterator>
where
    I::Item: Copy,
{
    iter: I,
    n: u32,
    cur: I::Item,
    count: u32,
}

// Note: if this is used then there is a performance hit as the
// inner loop cannot experience vectorization
//
// TODO: maybe faster to build the vec and then repeat it into
// the destination array?
impl<I: Iterator> Iterator for NTimesIter<I>
where
    I::Item: Copy,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count == 0 {
            self.count = self.n - 1;
            self.cur = self.iter.next()?;
        } else {
            self.count -= 1;
        }
        Some(self.cur)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();
        let lower = lower * self.n as usize;
        let upper = upper.map(|u| u * self.n as usize);
        (lower, upper)
    }
}

pub struct FnGen<T, ArrayType, F: FnMut(&mut rand_xoshiro::Xoshiro256PlusPlus) -> T>
where
    T: Copy + Default,
    ArrayType: arrow_array::Array + From<Vec<T>>,
{
    data_type: DataType,
    generator: F,
    array_type: PhantomData<ArrayType>,
    repeat: u32,
    leftover: T,
    leftover_count: u32,
    element_size_bytes: Option<ByteCount>,
}

impl<T, ArrayType, F: FnMut(&mut rand_xoshiro::Xoshiro256PlusPlus) -> T> FnGen<T, ArrayType, F>
where
    T: Copy + Default,
    ArrayType: arrow_array::Array + From<Vec<T>>,
{
    fn new_known_size(
        data_type: DataType,
        generator: F,
        repeat: u32,
        element_size_bytes: ByteCount,
    ) -> Self {
        Self {
            data_type,
            generator,
            array_type: PhantomData,
            repeat,
            leftover: T::default(),
            leftover_count: 0,
            element_size_bytes: Some(element_size_bytes),
        }
    }
}

impl<T, ArrayType, F: FnMut(&mut rand_xoshiro::Xoshiro256PlusPlus) -> T> ArrayGenerator
    for FnGen<T, ArrayType, F>
where
    T: Copy + Default,
    ArrayType: arrow_array::Array + From<Vec<T>> + 'static,
{
    fn generate(
        &mut self,
        length: RowCount,
        rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
    ) -> Result<Arc<dyn arrow_array::Array>, ArrowError> {
        let iter = (0..length.0).map(|_| (self.generator)(rng));
        let values = if self.repeat > 1 {
            Vec::from_iter(
                NTimesIter {
                    iter,
                    n: self.repeat,
                    cur: self.leftover,
                    count: self.leftover_count,
                }
                .take(length.0 as usize),
            )
        } else {
            Vec::from_iter(iter)
        };
        self.leftover_count = ((self.leftover_count as u64 + length.0) % self.repeat as u64) as u32;
        self.leftover = values.last().copied().unwrap_or(T::default());
        Ok(Arc::new(ArrayType::from(values)))
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn element_size_bytes(&self) -> Option<ByteCount> {
        self.element_size_bytes
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Seed(u64);
pub const DEFAULT_SEED: Seed = Seed(42);

impl From<u64> for Seed {
    fn from(n: u64) -> Self {
        Self(n)
    }
}

pub struct CycleVectorGenerator {
    underlying_gen: Box<dyn ArrayGenerator>,
    dimension: Dimension,
    data_type: DataType,
}

impl CycleVectorGenerator {
    pub fn new(underlying_gen: Box<dyn ArrayGenerator>, dimension: Dimension) -> Self {
        let data_type = DataType::FixedSizeList(
            Arc::new(Field::new("item", underlying_gen.data_type().clone(), true)),
            dimension.0 as i32,
        );
        Self {
            underlying_gen,
            dimension,
            data_type,
        }
    }
}

impl ArrayGenerator for CycleVectorGenerator {
    fn generate(
        &mut self,
        length: RowCount,
        rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
    ) -> Result<Arc<dyn arrow_array::Array>, ArrowError> {
        let values = self
            .underlying_gen
            .generate(RowCount::from(length.0 * self.dimension.0 as u64), rng)?;
        let field = Arc::new(Field::new("item", values.data_type().clone(), true));
        let values = Arc::new(values);

        let array = FixedSizeListArray::try_new(field, self.dimension.0 as i32, values, None)?;

        Ok(Arc::new(array))
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn element_size_bytes(&self) -> Option<ByteCount> {
        self.underlying_gen
            .element_size_bytes()
            .map(|byte_count| ByteCount::from(byte_count.0 * self.dimension.0 as u64))
    }
}

pub struct RandomBinaryGenerator {
    bytes_per_element: ByteCount,
    scale_to_utf8: bool,
    data_type: DataType,
}

impl RandomBinaryGenerator {
    pub fn new(bytes_per_element: ByteCount, scale_to_utf8: bool) -> Self {
        Self {
            bytes_per_element,
            scale_to_utf8,
            data_type: if scale_to_utf8 {
                Utf8Type::DATA_TYPE.clone()
            } else {
                BinaryType::DATA_TYPE.clone()
            },
        }
    }
}

impl ArrayGenerator for RandomBinaryGenerator {
    fn generate(
        &mut self,
        length: RowCount,
        rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
    ) -> Result<Arc<dyn arrow_array::Array>, ArrowError> {
        let mut bytes = vec![0; (self.bytes_per_element.0 * length.0) as usize];
        rng.fill_bytes(&mut bytes);
        if self.scale_to_utf8 {
            // This doesn't give us the full UTF-8 range and it isn't statistically correct but
            // it's fast and probably good enough for most cases
            bytes = bytes.into_iter().map(|val| (val % 95) + 32).collect();
        }
        let bytes = Buffer::from(bytes);
        let offsets = OffsetBuffer::from_lengths(
            iter::repeat(self.bytes_per_element.0 as usize).take(length.0 as usize),
        );
        if self.scale_to_utf8 {
            Ok(Arc::new(arrow_array::StringArray::new_unchecked(
                offsets, bytes, None,
            )))
        } else {
            Ok(Arc::new(arrow_array::BinaryArray::new(
                offsets, bytes, None,
            )))
        }
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn element_size_bytes(&self) -> Option<ByteCount> {
        // Not exactly correct since there are N + 1 4-byte offsets and this only counts N
        Some(ByteCount::from(
            self.bytes_per_element.0 + std::mem::size_of::<i32>() as u64,
        ))
    }
}

pub struct FixedBinaryGenerator<T: ByteArrayType> {
    value: Vec<u8>,
    data_type: DataType,
    array_type: PhantomData<T>,
}

impl<T: ByteArrayType> FixedBinaryGenerator<T> {
    pub fn new(value: Vec<u8>) -> Self {
        Self {
            value,
            data_type: T::DATA_TYPE.clone(),
            array_type: PhantomData,
        }
    }
}

impl<T: ByteArrayType> ArrayGenerator for FixedBinaryGenerator<T> {
    fn generate(
        &mut self,
        length: RowCount,
        _: &mut rand_xoshiro::Xoshiro256PlusPlus,
    ) -> Result<Arc<dyn arrow_array::Array>, ArrowError> {
        let bytes = Buffer::from(Vec::from_iter(
            self.value
                .iter()
                .cycle()
                .take((length.0 * self.value.len() as u64) as usize)
                .copied(),
        ));
        let offsets =
            OffsetBuffer::from_lengths(iter::repeat(self.value.len()).take(length.0 as usize));
        Ok(Arc::new(arrow_array::GenericByteArray::<T>::new(
            offsets, bytes, None,
        )))
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn element_size_bytes(&self) -> Option<ByteCount> {
        // Not exactly correct since there are N + 1 4-byte offsets and this only counts N
        Some(ByteCount::from(
            self.value.len() as u64 + std::mem::size_of::<i32>() as u64,
        ))
    }
}

pub struct DictionaryGenerator<K: ArrowDictionaryKeyType> {
    generator: Box<dyn ArrayGenerator>,
    data_type: DataType,
    key_type: PhantomData<K>,
    key_width: u64,
}

impl<K: ArrowDictionaryKeyType> DictionaryGenerator<K> {
    fn new(generator: Box<dyn ArrayGenerator>) -> Self {
        let key_type = Box::new(K::DATA_TYPE.clone());
        let key_width = key_type
            .primitive_width()
            .expect("dictionary key types should have a known width")
            as u64;
        let val_type = Box::new(generator.data_type().clone());
        let dict_type = DataType::Dictionary(key_type, val_type);
        Self {
            generator,
            data_type: dict_type,
            key_type: PhantomData,
            key_width,
        }
    }
}

impl<K: ArrowDictionaryKeyType> ArrayGenerator for DictionaryGenerator<K> {
    fn generate(
        &mut self,
        length: RowCount,
        rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
    ) -> Result<Arc<dyn arrow_array::Array>, ArrowError> {
        let underlying = self.generator.generate(length, rng)?;
        arrow_cast::cast::cast(&underlying, &self.data_type)
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn element_size_bytes(&self) -> Option<ByteCount> {
        self.generator
            .element_size_bytes()
            .map(|size_bytes| ByteCount::from(size_bytes.0 + self.key_width))
    }
}

/// A RecordBatchReader that generates batches of the given size from the given array generators
pub struct FixedSizeBatchGenerator {
    rng: rand_xoshiro::Xoshiro256PlusPlus,
    generators: Vec<Box<dyn ArrayGenerator>>,
    batch_size: RowCount,
    num_batches: BatchCount,
    schema: SchemaRef,
}

impl FixedSizeBatchGenerator {
    fn new(
        generators: Vec<(Option<String>, Box<dyn ArrayGenerator>)>,
        batch_size: RowCount,
        num_batches: BatchCount,
        seed: Option<Seed>,
        default_null_probability: Option<f64>,
    ) -> Self {
        let mut fields = Vec::with_capacity(generators.len());
        for (field_index, field_gen) in generators.iter().enumerate() {
            let (name, gen) = field_gen;
            let default_name = format!("field_{}", field_index);
            let name = name.clone().unwrap_or(default_name);
            fields.push(Field::new(name, gen.data_type().clone(), true));
        }
        let mut generators = generators
            .into_iter()
            .map(|(_, gen)| gen)
            .collect::<Vec<_>>();
        if let Some(null_probability) = default_null_probability {
            generators = generators
                .into_iter()
                .map(|gen| gen.with_nulls(null_probability))
                .collect();
        }
        let schema = Arc::new(Schema::new(fields));
        Self {
            rng: rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(
                seed.map(|s| s.0).unwrap_or(DEFAULT_SEED.0),
            ),
            generators,
            batch_size,
            num_batches,
            schema,
        }
    }

    fn gen_next(&mut self) -> Result<RecordBatch, ArrowError> {
        let mut arrays = Vec::with_capacity(self.generators.len());
        for gen in self.generators.iter_mut() {
            let arr = gen.generate(self.batch_size, &mut self.rng)?;
            arrays.push(arr);
        }
        self.num_batches.0 -= 1;
        Ok(RecordBatch::try_new(self.schema.clone(), arrays).unwrap())
    }
}

impl Iterator for FixedSizeBatchGenerator {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.num_batches.0 == 0 {
            return None;
        }
        Some(self.gen_next())
    }
}

impl RecordBatchReader for FixedSizeBatchGenerator {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// A builder to create a record batch reader with generated data
///
/// This type is meant to be used in a fluent builder style to define the schema and generators
/// for a record batch reader.
#[derive(Default)]
pub struct BatchGeneratorBuilder {
    generators: Vec<(Option<String>, Box<dyn ArrayGenerator>)>,
    default_null_probability: Option<f64>,
    seed: Option<Seed>,
}

pub enum RoundingBehavior {
    ExactOrErr,
    RoundUp,
    RoundDown,
}

impl BatchGeneratorBuilder {
    /// Create a new BatchGeneratorBuilder with a default random seed
    pub fn new() -> Self {
        Default::default()
    }

    /// Create a new BatchGeneratorBuilder with the given seed
    pub fn new_with_seed(seed: Seed) -> Self {
        Self {
            seed: Some(seed),
            ..Default::default()
        }
    }

    /// Adds a new column to the generator
    ///
    /// See [`crate::generator::array`] for methods to create generators
    pub fn col(mut self, name: Option<String>, gen: Box<dyn ArrayGenerator>) -> Self {
        self.generators.push((name, gen));
        self
    }

    /// Create a RecordBatchReader that generates batches of the given size (in rows)
    pub fn into_reader_rows(
        self,
        batch_size: RowCount,
        num_batches: BatchCount,
    ) -> impl RecordBatchReader {
        FixedSizeBatchGenerator::new(
            self.generators,
            batch_size,
            num_batches,
            self.seed,
            self.default_null_probability,
        )
    }

    /// Create a RecordBatchReader that generates batches of the given size (in bytes)
    pub fn into_reader_bytes(
        self,
        batch_size_bytes: ByteCount,
        num_batches: BatchCount,
        rounding: RoundingBehavior,
    ) -> Result<impl RecordBatchReader, ArrowError> {
        let bytes_per_row = self
            .generators
            .iter()
            .map(|gen| gen.1.element_size_bytes().map(|byte_count| byte_count.0).ok_or(
                        ArrowError::NotYetImplemented("The function into_reader_bytes currently requires each array generator to have a fixed element size".to_string())
                )
            )
            .sum::<Result<u64, ArrowError>>()?;
        let mut num_rows = RowCount::from(batch_size_bytes.0 / bytes_per_row);
        if batch_size_bytes.0 % bytes_per_row != 0 {
            match rounding {
                RoundingBehavior::ExactOrErr => {
                    return Err(ArrowError::NotYetImplemented(
                        format!("Exact rounding requested but not possible.  Batch size requested {}, row size: {}", batch_size_bytes.0, bytes_per_row))
                    );
                }
                RoundingBehavior::RoundUp => {
                    num_rows = RowCount::from(num_rows.0 + 1);
                }
                RoundingBehavior::RoundDown => (),
            }
        }
        Ok(self.into_reader_rows(num_rows, num_batches))
    }

    /// Set the seed for the generator
    pub fn with_seed(&mut self, seed: Seed) {
        self.seed = Some(seed);
    }

    /// Adds nulls (with the given probability) to all columns
    pub fn with_nulls(&mut self, default_null_probability: f64) {
        self.default_null_probability = Some(default_null_probability);
    }
}

const MS_PER_DAY: i64 = 86400000;

pub mod array {
    use arrow_array::types::{
        ArrowPrimitiveType, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type,
        UInt16Type, UInt32Type, UInt64Type, UInt8Type, Utf8Type,
    };
    use arrow_array::{ArrowNativeTypeOp, Date32Array, Date64Array, PrimitiveArray};
    use rand::distributions::Uniform;
    use rand::prelude::Distribution;
    use rand::Rng;

    use super::*;

    /// Create a generator of vectors by continuously calling the given generator
    ///
    /// For example, given a step generator and a dimension of 3 this will generate vectors like
    /// [0, 1, 2], [3, 4, 5], [6, 7, 8], ...
    pub fn cycle_vec(
        generator: Box<dyn ArrayGenerator>,
        dimension: Dimension,
    ) -> Box<dyn ArrayGenerator> {
        Box::new(CycleVectorGenerator::new(generator, dimension))
    }

    /// Create a generator that starts at 0 and increments by 1 for each element
    pub fn step<DataType>() -> Box<dyn ArrayGenerator>
    where
        DataType::Native: Copy + Default + std::ops::AddAssign<DataType::Native> + 'static,
        DataType: ArrowPrimitiveType,
        PrimitiveArray<DataType>: From<Vec<DataType::Native>> + 'static,
    {
        let mut x = DataType::Native::default();
        Box::new(
            FnGen::<DataType::Native, PrimitiveArray<DataType>, _>::new_known_size(
                DataType::DATA_TYPE.clone(),
                move |_| {
                    let y = x;
                    x += DataType::Native::ONE;
                    y
                },
                1,
                DataType::DATA_TYPE
                    .primitive_width()
                    .map(|width| ByteCount::from(width as u64))
                    .expect("Primitive types should have a fixed width"),
            ),
        )
    }

    /// Create a generator that starts at a given value and increments by a given step for each element
    pub fn step_custom<DataType>(
        start: DataType::Native,
        step: DataType::Native,
    ) -> Box<dyn ArrayGenerator>
    where
        DataType::Native: Copy + Default + std::ops::AddAssign<DataType::Native> + 'static,
        PrimitiveArray<DataType>: From<Vec<DataType::Native>> + 'static,
        DataType: ArrowPrimitiveType,
    {
        let mut x = start;
        Box::new(
            FnGen::<DataType::Native, PrimitiveArray<DataType>, _>::new_known_size(
                DataType::DATA_TYPE.clone(),
                move |_| {
                    let y = x;
                    x += step;
                    y
                },
                1,
                DataType::DATA_TYPE
                    .primitive_width()
                    .map(|width| ByteCount::from(width as u64))
                    .expect("Primitive types should have a fixed width"),
            ),
        )
    }

    /// Create a generator that fills each element with the given primitive value
    pub fn fill<DataType>(value: DataType::Native) -> Box<dyn ArrayGenerator>
    where
        DataType::Native: Copy + 'static,
        DataType: ArrowPrimitiveType,
        PrimitiveArray<DataType>: From<Vec<DataType::Native>> + 'static,
    {
        Box::new(
            FnGen::<DataType::Native, PrimitiveArray<DataType>, _>::new_known_size(
                DataType::DATA_TYPE.clone(),
                move |_| value,
                1,
                DataType::DATA_TYPE
                    .primitive_width()
                    .map(|width| ByteCount::from(width as u64))
                    .expect("Primitive types should have a fixed width"),
            ),
        )
    }

    /// Create a generator that fills each element with the given binary value
    pub fn fill_varbin(value: Vec<u8>) -> Box<dyn ArrayGenerator> {
        Box::new(FixedBinaryGenerator::<BinaryType>::new(value))
    }

    /// Create a generator that fills each element with the given string value
    pub fn fill_utf8(value: String) -> Box<dyn ArrayGenerator> {
        Box::new(FixedBinaryGenerator::<Utf8Type>::new(value.into_bytes()))
    }

    /// Create a generator of primitive values that are randomly sampled from the entire range available for the value
    pub fn rand<DataType>() -> Box<dyn ArrayGenerator>
    where
        DataType::Native: Copy + 'static,
        PrimitiveArray<DataType>: From<Vec<DataType::Native>> + 'static,
        DataType: ArrowPrimitiveType,
        rand::distributions::Standard: rand::distributions::Distribution<DataType::Native>,
    {
        Box::new(
            FnGen::<DataType::Native, PrimitiveArray<DataType>, _>::new_known_size(
                DataType::DATA_TYPE.clone(),
                move |rng| rng.gen(),
                1,
                DataType::DATA_TYPE
                    .primitive_width()
                    .map(|width| ByteCount::from(width as u64))
                    .expect("Primitive types should have a fixed width"),
            ),
        )
    }

    /// Create a generator of 1d vectors (of a primitive type) consisting of randomly sampled primitive values
    pub fn rand_vec<DataType>(dimension: Dimension) -> Box<dyn ArrayGenerator>
    where
        DataType::Native: Copy + 'static,
        PrimitiveArray<DataType>: From<Vec<DataType::Native>> + 'static,
        DataType: ArrowPrimitiveType,
        rand::distributions::Standard: rand::distributions::Distribution<DataType::Native>,
    {
        let underlying = rand::<DataType>();
        cycle_vec(underlying, dimension)
    }

    /// Create a generator of randomly sampled date32 values
    ///
    /// Instead of sampling the entire range, all values will be drawn from the last year as this
    /// is a more common use pattern
    pub fn rand_date32() -> Box<dyn ArrayGenerator> {
        let data_type = DataType::Date32;
        let now_ms = chrono::Utc::now().timestamp_millis();
        let now_days = (now_ms / MS_PER_DAY) as i32;
        let one_year_ago = now_days - 365;
        let dist = Uniform::new(one_year_ago, now_days);

        Box::new(FnGen::<i32, Date32Array, _>::new_known_size(
            data_type.clone(),
            move |rng| dist.sample(rng),
            1,
            DataType::Date32
                .primitive_width()
                .map(|width| ByteCount::from(width as u64))
                .expect("Date32 should have a fixed width"),
        ))
    }

    /// Create a generator of randomly sampled date64 values
    ///
    /// Instead of sampling the entire range, all values will be drawn from the last year as this
    /// is a more common use pattern
    pub fn rand_date64() -> Box<dyn ArrayGenerator> {
        let data_type = DataType::Date64;
        let now_ms = chrono::Utc::now().timestamp_millis();
        let now_days = (now_ms / MS_PER_DAY) as i32;
        let one_year_ago = now_days - 365;
        let dist = Uniform::new(one_year_ago, now_days);

        Box::new(FnGen::<i64, Date64Array, _>::new_known_size(
            data_type.clone(),
            move |rng| (dist.sample(rng) as i64) * MS_PER_DAY,
            1,
            DataType::Date64
                .primitive_width()
                .map(|width| ByteCount::from(width as u64))
                .expect("Date64 should have a fixed width"),
        ))
    }

    /// Create a generator of random binary values
    pub fn rand_varbin(bytes_per_element: ByteCount) -> Box<dyn ArrayGenerator> {
        Box::new(RandomBinaryGenerator::new(bytes_per_element, false))
    }

    /// Create a generator of random strings
    ///
    /// All strings will consist entirely of printable ASCII characters
    pub fn rand_utf8(bytes_per_element: ByteCount) -> Box<dyn ArrayGenerator> {
        Box::new(RandomBinaryGenerator::new(bytes_per_element, true))
    }

    /// Create a generator of random values
    pub fn rand_type(data_type: &DataType) -> Box<dyn ArrayGenerator> {
        match data_type {
            DataType::Int8 => rand::<Int8Type>(),
            DataType::Int16 => rand::<Int16Type>(),
            DataType::Int32 => rand::<Int32Type>(),
            DataType::Int64 => rand::<Int64Type>(),
            DataType::UInt8 => rand::<UInt8Type>(),
            DataType::UInt16 => rand::<UInt16Type>(),
            DataType::UInt32 => rand::<UInt32Type>(),
            DataType::UInt64 => rand::<UInt64Type>(),
            DataType::Float32 => rand::<Float32Type>(),
            DataType::Float64 => rand::<Float64Type>(),
            DataType::Utf8 => rand_utf8(ByteCount::from(12)),
            DataType::Binary => rand_varbin(ByteCount::from(12)),
            DataType::Dictionary(key_type, value_type) => {
                dict_type(rand_type(value_type), key_type)
            }
            DataType::FixedSizeList(child, dimension) => cycle_vec(
                rand_type(child.data_type()),
                Dimension::from(*dimension as u32),
            ),
            DataType::Date32 => rand_date32(),
            DataType::Date64 => rand_date64(),
            _ => unimplemented!(),
        }
    }

    /// Encodes arrays generated by the underlying generator as dictionaries with the given key type
    ///
    /// Note that this may not be very realistic if the underlying generator is something like a random
    /// generator since most of the underlying values will be unique and the common case for dictionary
    /// encoding is when there is a small set of possible values.
    pub fn dict<K: ArrowDictionaryKeyType>(
        generator: Box<dyn ArrayGenerator>,
    ) -> Box<dyn ArrayGenerator> {
        Box::new(DictionaryGenerator::<K>::new(generator))
    }

    /// Encodes arrays generated by the underlying generator as dictionaries with the given key type
    pub fn dict_type(
        generator: Box<dyn ArrayGenerator>,
        key_type: &DataType,
    ) -> Box<dyn ArrayGenerator> {
        match key_type {
            DataType::Int8 => dict::<Int8Type>(generator),
            DataType::Int16 => dict::<Int16Type>(generator),
            DataType::Int32 => dict::<Int32Type>(generator),
            DataType::Int64 => dict::<Int64Type>(generator),
            DataType::UInt8 => dict::<UInt8Type>(generator),
            DataType::UInt16 => dict::<UInt16Type>(generator),
            DataType::UInt32 => dict::<UInt32Type>(generator),
            DataType::UInt64 => dict::<UInt64Type>(generator),
            _ => unimplemented!(),
        }
    }
}

/// Create a BatchGeneratorBuilder to start generating data
pub fn gen() -> BatchGeneratorBuilder {
    BatchGeneratorBuilder::default()
}

/// Create a BatchGeneratorBuilder with the given schema
///
/// You can add more columns or convert this into a reader immediately
pub fn rand(schema: &Schema) -> BatchGeneratorBuilder {
    let mut builder = BatchGeneratorBuilder::default();
    for field in schema.fields() {
        builder = builder.col(
            Some(field.name().clone()),
            array::rand_type(field.data_type()),
        );
    }
    builder
}

#[cfg(test)]
mod tests {

    use arrow_array::{
        types::{Float32Type, Int16Type, Int32Type, Int8Type},
        Float32Array, Int16Array, Int32Array, Int8Array,
    };

    use super::*;

    #[test]
    fn test_step() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(DEFAULT_SEED.0);
        let mut gen = array::step::<Int32Type>();
        assert_eq!(
            *gen.generate(RowCount::from(5), &mut rng).unwrap(),
            Int32Array::from_iter([0, 1, 2, 3, 4])
        );
        assert_eq!(
            *gen.generate(RowCount::from(5), &mut rng).unwrap(),
            Int32Array::from_iter([5, 6, 7, 8, 9])
        );

        let mut gen = array::step::<Int8Type>();
        assert_eq!(
            *gen.generate(RowCount::from(3), &mut rng).unwrap(),
            Int8Array::from_iter([0, 1, 2])
        );

        let mut gen = array::step::<Float32Type>();
        assert_eq!(
            *gen.generate(RowCount::from(3), &mut rng).unwrap(),
            Float32Array::from_iter([0.0, 1.0, 2.0])
        );

        let mut gen = array::step_custom::<Int16Type>(4, 8);
        assert_eq!(
            *gen.generate(RowCount::from(3), &mut rng).unwrap(),
            Int16Array::from_iter([4, 12, 20])
        );
        assert_eq!(
            *gen.generate(RowCount::from(2), &mut rng).unwrap(),
            Int16Array::from_iter([28, 36])
        );
    }

    #[test]
    fn test_fill() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(DEFAULT_SEED.0);
        let mut gen = array::fill::<Int32Type>(42);
        assert_eq!(
            *gen.generate(RowCount::from(3), &mut rng).unwrap(),
            Int32Array::from_iter([42, 42, 42])
        );
        assert_eq!(
            *gen.generate(RowCount::from(3), &mut rng).unwrap(),
            Int32Array::from_iter([42, 42, 42])
        );

        let mut gen = array::fill_varbin(vec![0, 1, 2]);
        assert_eq!(
            *gen.generate(RowCount::from(3), &mut rng).unwrap(),
            arrow_array::BinaryArray::from_iter_values([
                "\x00\x01\x02",
                "\x00\x01\x02",
                "\x00\x01\x02"
            ])
        );

        let mut gen = array::fill_utf8("xyz".to_string());
        assert_eq!(
            *gen.generate(RowCount::from(3), &mut rng).unwrap(),
            arrow_array::StringArray::from_iter_values(["xyz", "xyz", "xyz"])
        );
    }

    #[test]
    fn test_rng() {
        // Note: these tests are heavily dependent on the default seed.
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(DEFAULT_SEED.0);
        let mut gen = array::rand::<Int32Type>();
        assert_eq!(
            *gen.generate(RowCount::from(3), &mut rng).unwrap(),
            Int32Array::from_iter([-797553329, 1369325940, -69174021])
        );

        let mut gen = array::rand_varbin(ByteCount::from(3));
        assert_eq!(
            *gen.generate(RowCount::from(3), &mut rng).unwrap(),
            arrow_array::BinaryArray::from_iter_values([
                [184, 53, 216],
                [12, 96, 159],
                [125, 179, 56]
            ])
        );

        let mut gen = array::rand_utf8(ByteCount::from(3));
        assert_eq!(
            *gen.generate(RowCount::from(3), &mut rng).unwrap(),
            arrow_array::StringArray::from_iter_values([">@p", "n `", "NWa"])
        );

        let mut gen = array::rand_date32();
        let days_32 = gen.generate(RowCount::from(3), &mut rng).unwrap();
        assert_eq!(days_32.data_type(), &DataType::Date32);

        let mut gen = array::rand_date64();
        let days_64 = gen.generate(RowCount::from(3), &mut rng).unwrap();
        assert_eq!(days_64.data_type(), &DataType::Date64);
    }

    #[test]
    fn test_nulls() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(DEFAULT_SEED.0);
        let mut gen = array::rand::<Int32Type>().with_nulls(0.3);

        let arr = gen.generate(RowCount::from(1000), &mut rng).unwrap();

        // This assert depends on the default seed
        assert_eq!(arr.null_count(), 297);

        for len in 0..100 {
            let arr = gen.generate(RowCount::from(len), &mut rng).unwrap();
            // Make sure the null count we came up with matches the actual # of unset bits
            assert_eq!(
                arr.null_count(),
                arr.nulls()
                    .map(|nulls| (len as usize)
                        - nulls.buffer().count_set_bits_offset(0, len as usize))
                    .unwrap_or(0)
            );
        }

        let mut gen = array::rand::<Int32Type>().with_nulls(0.0);
        let arr = gen.generate(RowCount::from(10), &mut rng).unwrap();

        assert_eq!(arr.null_count(), 0);

        let mut gen = array::rand::<Int32Type>().with_nulls(1.0);
        let arr = gen.generate(RowCount::from(10), &mut rng).unwrap();

        assert_eq!(arr.null_count(), 10);
        assert!((0..10).all(|idx| arr.is_null(idx)));
    }

    #[test]
    fn test_rand_schema() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8, true),
            Field::new("c", DataType::Float32, true),
            Field::new("d", DataType::Int32, true),
            Field::new("e", DataType::Int32, true),
        ]);
        let rbr = rand(&schema)
            .into_reader_bytes(
                ByteCount::from(1024 * 1024),
                BatchCount::from(8),
                RoundingBehavior::ExactOrErr,
            )
            .unwrap();
        assert_eq!(*rbr.schema(), schema);

        let batches = rbr.map(|val| val.unwrap()).collect::<Vec<_>>();
        assert_eq!(batches.len(), 8);

        for batch in batches {
            assert_eq!(batch.num_rows(), 1024 * 1024 / 32);
            assert_eq!(batch.num_columns(), 5);
        }
    }
}
