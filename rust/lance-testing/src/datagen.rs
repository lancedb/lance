// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Data generation utilities for unit tests

use std::iter::repeat_with;
use std::sync::Arc;

use crate::{
    arrow::{fixed_size_list_type, FixedSizeListArrayExt},
};

use arrow_array::{
    ArrowNumericType, Float32Array, Int32Array, NativeAdapter, PrimitiveArray, RecordBatch,
    RecordBatchIterator, RecordBatchReader,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use num_traits::{real::Real, FromPrimitive};
use rand::{Rng, SeedableRng, rngs::StdRng};

pub trait ArrayGenerator {
    fn generate(&mut self, length: usize) -> Arc<dyn arrow_array::Array>;
    fn data_type(&self) -> &DataType;
    fn name(&self) -> Option<&str>;
}

pub struct IncrementingInt32 {
    name: Option<String>,
    current: i32,
    step: i32,
}

impl Default for IncrementingInt32 {
    fn default() -> Self {
        Self {
            name: None,
            current: 0,
            step: 1,
        }
    }
}

impl IncrementingInt32 {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn start(mut self, start: i32) -> Self {
        self.current = start;
        self
    }

    pub fn step(mut self, step: i32) -> Self {
        self.step = step;
        self
    }

    pub fn named(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }
}

impl ArrayGenerator for IncrementingInt32 {
    fn generate(&mut self, length: usize) -> Arc<dyn arrow_array::Array> {
        let mut values = Vec::with_capacity(length);
        for _ in 0..length {
            values.push(self.current);
            self.current += self.step;
        }
        Arc::new(Int32Array::from(values))
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn data_type(&self) -> &DataType {
        &DataType::Int32
    }
}

pub struct RandomVector {
    name: Option<String>,
    vec_width: i32,
    data_type: DataType,
}

impl Default for RandomVector {
    fn default() -> Self {
        Self {
            name: None,
            vec_width: 4,
            data_type: fixed_size_list_type(4, DataType::Float32),
        }
    }
}

impl RandomVector {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn vec_width(mut self, vec_width: i32) -> Self {
        self.vec_width = vec_width;
        self.data_type = fixed_size_list_type(self.vec_width, DataType::Float32);
        self
    }

    pub fn named(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }
}

impl ArrayGenerator for RandomVector {
    fn generate(&mut self, length: usize) -> Arc<dyn arrow_array::Array> {
        let values = generate_random_array(length * (self.vec_width as usize));
        Ok(Arc::new(
            <arrow_array::FixedSizeListArray as FixedSizeListArrayExt>::try_new_from_values(
                values,
                self.vec_width,
            )?,
        ))
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }
}

#[derive(Default)]
pub struct BatchGenerator {
    generators: Vec<Box<dyn ArrayGenerator>>,
}

impl BatchGenerator {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn col(mut self, gen: Box<dyn ArrayGenerator>) -> Self {
        self.generators.push(gen);
        self
    }

    pub fn batch(&mut self, num_rows: i32) -> impl RecordBatchReader {
        let mut fields = Vec::with_capacity(self.generators.len());
        let mut arrays = Vec::with_capacity(self.generators.len());
        for (field_index, gen) in self.generators.iter_mut().enumerate() {
            let arr = gen.generate(num_rows as usize);
            let default_name = format!("field_{}", field_index);
            let name = gen.name().unwrap_or(&default_name);
            fields.push(Field::new(name, arr.data_type().clone(), true));
            arrays.push(arr);
        }
        let schema = Arc::new(ArrowSchema::new(fields));
        let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();
        RecordBatchIterator::new(
            vec![batch].into_iter().map(Ok),
            schema.clone(),
        )
    }
}

/// Returns a batch of data that has a column that can be used to create an ANN index
///
/// The indexable column will be named "indexable"
/// The batch will not be empty
/// There will only be one batch
///
/// There are no other assumptions it is safe to make about the returned reader
pub fn some_indexable_batch() -> impl RecordBatchReader {
    let x = Box::new(RandomVector::new().named("indexable".to_string()));
    BatchGenerator::new().col(x).batch(512)
}

/// Returns a non-empty batch of data
///
/// The batch will not be empty
/// There will only be one batch
///
/// There are no other assumptions it is safe to make about the returned reader
pub fn some_batch() -> impl RecordBatchReader {
    some_indexable_batch()
}

/// Create a random float32 array.
pub fn generate_random_array_with_seed<T: ArrowNumericType>(
    n: usize,
    seed: [u8; 32],
) -> PrimitiveArray<T>
where
    T::Native: Real + FromPrimitive,
    NativeAdapter<T>: From<T::Native>,
{
    let mut rng = StdRng::from_seed(seed);

    PrimitiveArray::<T>::from_iter(repeat_with(|| T::Native::from_f32(rng.gen::<f32>())).take(n))
}

/// Create a random float32 array.
pub fn generate_random_array(n: usize) -> Float32Array {
    let mut rng = rand::thread_rng();
    Float32Array::from(
        repeat_with(|| rng.gen::<f32>())
            .take(n)
            .collect::<Vec<f32>>(),
    )
}
