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

using arrow::Int32Builder;

TEST_CASE("Test Write Int32 array") {
  Int32Builder builder;
  CHECK(builder.AppendValues({1, 2, 3, 4, 5, 6, 7, 8}).ok());
  auto arr = builder.Finish().ValueOrDie();
  CHECK(arr->length() == 8);

  auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
  lance::encodings::PlainEncoder encoder(sink);
  auto offset = encoder.Write(arr).ValueOrDie();

  // Read it back
  auto infile = make_shared<arrow::io::BufferReader>(sink->Finish().ValueOrDie());
  lance::encodings::PlainDecoder<::arrow::Int32Type> decoder(infile, offset, arr->length());
  auto actual = decoder.ToArray().ValueOrDie();
  CHECK(arr->Equals(actual));

  for (int i = 0; i < arr->length(); i++) {
    CHECK(arr->GetScalar(i).ValueOrDie()->Equals(decoder.GetScalar(i).ValueOrDie()));
  }
}