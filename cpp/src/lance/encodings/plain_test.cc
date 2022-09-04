//  Copyright 2022 Lance Authors
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "lance/encodings/plain.h"

#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/io/api.h>

#include <catch2/catch_test_macros.hpp>

#include "lance/arrow/stl.h"

using arrow::Int32Builder;
using lance::arrow::ToArray;

TEST_CASE("Test Write Int32 array") {
  auto arr = lance::arrow::ToArray({1, 2, 3, 4, 5, 6, 7, 8}).ValueOrDie();
  CHECK(arr->length() == 8);

  auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
  lance::encodings::PlainEncoder encoder(sink);
  auto offset = encoder.Write(arr).ValueOrDie();

  // Read it back
  auto infile = make_shared<arrow::io::BufferReader>(sink->Finish().ValueOrDie());
  lance::encodings::PlainDecoder decoder(infile, arrow::int32());
  CHECK(decoder.Init().ok());
  decoder.Reset(offset, arr->length());
  auto actual = decoder.ToArray().ValueOrDie();
  CHECK(arr->Equals(actual));

  for (int i = 0; i < arr->length(); i++) {
    CHECK(arr->GetScalar(i).ValueOrDie()->Equals(decoder.GetScalar(i).ValueOrDie()));
  }
}

TEST_CASE("Do not read full batch") {
  auto arr = lance::arrow::ToArray({10, 20, 30, 40, 50, 60, 70, 80}).ValueOrDie();
  CHECK(arr->length() == 8);

  auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
  lance::encodings::PlainEncoder encoder(sink);
  auto offset = encoder.Write(arr).ValueOrDie();

  // Read it back
  auto infile = make_shared<arrow::io::BufferReader>(sink->Finish().ValueOrDie());
  lance::encodings::PlainDecoder decoder(infile, arrow::int32());
  CHECK(decoder.Init().ok());
  decoder.Reset(offset, arr->length());

  // Decode arr[6:8]
  auto values = decoder.ToArray(6, 8).ValueOrDie();
  CHECK(values->Equals(lance::arrow::ToArray({70, 80}).ValueOrDie()));
}

TEST_CASE("Test take plain values") {
  arrow::Int32Builder builder;
  for (int i = 0; i < 100; i++) {
    CHECK(builder.Append(i).ok());
  }
  auto arr = builder.Finish().ValueOrDie();

  auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
  lance::encodings::PlainEncoder encoder(sink);
  auto offset = encoder.Write(arr).ValueOrDie();

  auto infile = make_shared<arrow::io::BufferReader>(sink->Finish().ValueOrDie());
  lance::encodings::PlainDecoder decoder(infile, arr->type());
  CHECK(decoder.Init().ok());
  decoder.Reset(offset, arr->length());

  auto indices = lance::arrow::ToArray({8, 12, 16, 20, 45}).ValueOrDie();
  auto actual = decoder.Take(indices).ValueOrDie();
  INFO("Indices " << indices->ToString() << " Actual " << actual->ToString());
  CHECK(actual->Equals(indices));
}

TEST_CASE("Write boolean array") {
  arrow::BooleanBuilder builder;
  for (int i = 0; i < 10; i++) {
    CHECK(builder.Append(i % 3 == 0).ok());
  }
  auto arr = builder.Finish().ValueOrDie();

  auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
  lance::encodings::PlainEncoder encoder(sink);
  auto offset = encoder.Write(arr).ValueOrDie();

  auto infile = make_shared<arrow::io::BufferReader>(sink->Finish().ValueOrDie());
  lance::encodings::PlainDecoder decoder(infile, arr->type());
  CHECK(decoder.Init().ok());
  decoder.Reset(offset, arr->length());

  auto actual = decoder.ToArray().ValueOrDie();
  CHECK(arr->Equals(actual));

  for (int i = 0; i < arr->length(); i++) {
    INFO("Expected: " << arr->GetScalar(i).ValueOrDie()->ToString()
                      << " Got: " << decoder.GetScalar(i).ValueOrDie()->ToString());
    CHECK(arr->GetScalar(i).ValueOrDie()->Equals(decoder.GetScalar(i).ValueOrDie()));
  }

  auto indices = ToArray({0, 3, 6}).ValueOrDie();
  CHECK(lance::arrow::ToArray({true, true, true})
            .ValueOrDie()
            ->Equals(decoder.Take(indices).ValueOrDie()));
}

void TestWriteFixedSizeArray(const std::shared_ptr<::arrow::Array>& arr) {
  auto out = arrow::io::BufferOutputStream::Create().ValueOrDie();

  auto encoder = ::lance::encodings::PlainEncoder(out);
  auto offset = encoder.Write(arr).ValueOrDie();

  auto infile = make_shared<arrow::io::BufferReader>(out->Finish().ValueOrDie());
  auto decoder = lance::encodings::PlainDecoder(infile, arr->type());
  CHECK(decoder.Init().ok());
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
