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

TEST_CASE("Write fixed size binary") {
  auto dtype = ::arrow::fixed_size_binary(10);
  auto builder = ::arrow::FixedSizeBinaryBuilder(dtype);
  CHECK(builder.Append("12345689").ok());
  auto arr = builder.Finish().ValueOrDie();

  auto out = arrow::io::BufferOutputStream::Create().ValueOrDie();

  auto encoder = ::lance::encodings::FixedSizeBinaryEncoder(out);
  CHECK(encoder.Write(arr).ok());
}

TEST_CASE("Write fixed size list") {
  auto dtype = ::arrow::fixed_size_list(::arrow::int32(), 4);
  auto int_builder = std::make_shared<::arrow::Int32Builder>();
  auto builder = ::arrow::FixedSizeListBuilder(::arrow::default_memory_pool(), int_builder, dtype);
  CHECK(builder.Append().ok());
  CHECK(int_builder->AppendValues({1, 2, 3, 4}).ok());
  CHECK(builder.Append().ok());
  CHECK(int_builder->AppendValues({5, 6, 7, 8}).ok());
  auto arr = builder.Finish().ValueOrDie();

  fmt::print("Arr {}\n", arr->ToString());

  auto out = arrow::io::BufferOutputStream::Create().ValueOrDie();

  auto encoder = ::lance::encodings::FixedSizeBinaryEncoder(out);
  CHECK(encoder.Write(arr).ok());
}
