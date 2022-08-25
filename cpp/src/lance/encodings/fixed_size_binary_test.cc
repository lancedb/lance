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

#include "lance/encodings/fixed_size_binary.h"

#include <arrow/builder.h>
#include <arrow/type.h>
#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>

#include "lance/arrow/stl.h"

void TestWriteFixedSizeArray(const std::shared_ptr<::arrow::Array>& arr) {
  auto out = arrow::io::BufferOutputStream::Create().ValueOrDie();

  auto encoder = ::lance::encodings::FixedSizeBinaryEncoder(out);
  auto offset = encoder.Write(arr).ValueOrDie();

  auto infile = make_shared<arrow::io::BufferReader>(out->Finish().ValueOrDie());
  auto decoder = lance::encodings::FixedSizedBinaryDecoder(infile, arr->type());
  decoder.Reset(offset, arr->length());

  auto actual = decoder.ToArray(0).ValueOrDie();
  INFO("Expected: " << arr->ToString() << "\nActual: " << actual->ToString());
  CHECK(arr->Equals(actual));

  auto indices = lance::arrow::ToArray({1, 3, 5}).ValueOrDie();
  auto values = decoder.Take(indices).ValueOrDie();
  for (int i = 0; i < indices->length(); i++) {
    CHECK(
        values->GetScalar(i).ValueOrDie()->Equals(arr->GetScalar(indices->Value(i)).ValueOrDie()));
  }
}

TEST_CASE("Write fixed size binary") {
  auto dtype = ::arrow::fixed_size_binary(10);
  auto builder = ::arrow::FixedSizeBinaryBuilder(dtype);
  for (int i = 0; i < 10; i++) {
    CHECK(builder.Append(fmt::format("number-{}", i)).ok());
  }
  auto arr = builder.Finish().ValueOrDie();

  TestWriteFixedSizeArray(arr);
}

TEST_CASE("Write fixed size list") {
  auto list_size = 4;
  auto dtype = ::arrow::fixed_size_list(::arrow::int32(), list_size);
  auto int_builder = std::make_shared<::arrow::Int32Builder>();
  auto builder = ::arrow::FixedSizeListBuilder(::arrow::default_memory_pool(), int_builder, dtype);

  for (int i = 0; i < 10; i++) {
    CHECK(builder.Append().ok());
    for (int j = 0; j < list_size; j++) {
      CHECK(int_builder->Append(i * list_size + j).ok());
    }
  }
  auto arr = builder.Finish().ValueOrDie();

  TestWriteFixedSizeArray(arr);
}
