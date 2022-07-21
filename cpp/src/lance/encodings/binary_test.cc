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

#include "lance/encodings/binary.h"

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/io/api.h>
#include <fmt/format.h>

#include <catch2/catch_test_macros.hpp>
#include <numeric>
#include <string>
#include <vector>

#include "lance/arrow/stl.h"
#include "lance/format/format.h"

using arrow::BinaryBuilder;
using arrow::Result;
using arrow::Status;
using arrow::StringArray;
using lance::encodings::VarBinaryDecoder;
using lance::encodings::VarBinaryEncoder;
using std::string;
using std::vector;


auto WriteStrings(std::shared_ptr<arrow::io::BufferOutputStream> out,
                  std::shared_ptr<StringArray> arr) {
  VarBinaryEncoder writer(out);
  return writer.Write(std::static_pointer_cast<arrow::StringArray>(arr)).ValueOrDie();
}

TEST_CASE("Write binary arrow") {
  auto out = arrow::io::BufferOutputStream::Create().ValueOrDie();

  auto arr1 = lance::arrow::ToArray({"First", "Second", "More"}).ValueOrDie();
  auto arr2 = lance::arrow::ToArray({"THIS", "IS", "SOMETHING", "ELSE"}).ValueOrDie();

  auto offset1 = WriteStrings(out, arr1);
  auto offset2 = WriteStrings(out, arr2);

  auto buf = out->Finish().ValueOrDie();
  auto infile = make_shared<arrow::io::BufferReader>(buf);

  {
    VarBinaryDecoder<::arrow::StringType> decoder(infile, arrow::utf8());
    decoder.Reset(offset1, 3);

    auto actual_arr = decoder.ToArray().ValueOrDie();
    CHECK(arr1->Equals(actual_arr));

    for (int64_t i = 0; i < arr1->length(); i++) {
      auto expected = decoder.GetScalar(i).ValueOrDie();
      CHECK(expected->CastTo(arrow::utf8()).ValueOrDie()->Equals(arr1->GetScalar(i).ValueOrDie()));
    }
  }

  {
    VarBinaryDecoder<::arrow::StringType> decoder(infile, arrow::utf8());
    decoder.Reset(offset2, 4);

    auto actual_arr = decoder.ToArray().ValueOrDie();
    INFO("ACTUAL ARR 2 " << actual_arr->ToString());
    CHECK(arr2->Equals(actual_arr));

    for (int64_t i = 0; i < arr1->length(); i++) {
      auto expected = decoder.GetScalar(i).ValueOrDie();
      CHECK(expected->CastTo(arrow::utf8()).ValueOrDie()->Equals(arr2->GetScalar(i).ValueOrDie()));
    }
  }
}

TEST_CASE("Take") {
  auto out = arrow::io::BufferOutputStream::Create().ValueOrDie();

  std::vector<std::string> words;
  for (int i = 0; i < 100; i++) {
    words.emplace_back(fmt::format("{}", i));
  }
  auto arr = lance::arrow::ToArray(words).ValueOrDie();
  auto offset = WriteStrings(out, arr);
  auto buf = out->Finish().ValueOrDie();
  auto infile = make_shared<arrow::io::BufferReader>(buf);

  VarBinaryDecoder<::arrow::StringType> decoder(infile, ::arrow::utf8());
  decoder.Reset(offset, 100);
  auto indices = lance::arrow::ToArray({5, 10, 20}).ValueOrDie();
  auto actual = decoder.Take(indices).ValueOrDie();
  auto expected = lance::arrow::ToArray({"5", "10", "20"}).ValueOrDie();
  CHECK(expected->Equals(actual));
}