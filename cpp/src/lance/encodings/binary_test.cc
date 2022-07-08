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

#include "lance/format/format.h"

using arrow::BinaryBuilder;
using arrow::Result;
using arrow::Status;
using arrow::StringArray;
using lance::encodings::VarBinaryDecoder;
using lance::encodings::VarBinaryEncoder;
using std::string;
using std::vector;

auto BuildStringArray(const vector<string>& words) {
  arrow::StringBuilder builder;
  for (auto s : words) {
    CHECK(builder.Append(s).ok());
  }
  return std::static_pointer_cast<arrow::StringArray>(builder.Finish().ValueOrDie());
}

auto WriteStrings(std::shared_ptr<arrow::io::BufferOutputStream> out,
                  std::shared_ptr<StringArray> arr) {
  VarBinaryEncoder writer(out);
  return writer.Write(std::static_pointer_cast<arrow::StringArray>(arr)).ValueOrDie();
}

TEST_CASE("Write binary arrow") {
  auto out = arrow::io::BufferOutputStream::Create().ValueOrDie();

  auto arr1 = BuildStringArray({"First", "Second", "More"});
  auto arr2 = BuildStringArray({"THIS", "IS", "SOMETHING", "ELSE"});

  auto offset1 = WriteStrings(out, arr1);
  auto offset2 = WriteStrings(out, arr2);

  auto buf = out->Finish().ValueOrDie();
  auto infile = make_shared<arrow::io::BufferReader>(buf);

  {
    VarBinaryDecoder<::arrow::StringType> decoder(infile, offset1, 3);

    auto actual_arr = decoder.ToArray().ValueOrDie();
    CHECK(arr1->Equals(actual_arr));

    for (int64_t i = 0; i < arr1->length(); i++) {
      auto expected = decoder.GetScalar(i).ValueOrDie();
      CHECK(expected->CastTo(arrow::utf8()).ValueOrDie()->Equals(arr1->GetScalar(i).ValueOrDie()));
    }
  }

  {
    VarBinaryDecoder<::arrow::StringType> decoder(infile, offset2, 4);

    auto actual_arr = decoder.ToArray().ValueOrDie();
    INFO("ACTUAL ARR 2 " << actual_arr->ToString());
    CHECK(arr2->Equals(actual_arr));

    for (int64_t i = 0; i < arr1->length(); i++) {
      auto expected = decoder.GetScalar(i).ValueOrDie();
      CHECK(expected->CastTo(arrow::utf8()).ValueOrDie()->Equals(arr2->GetScalar(i).ValueOrDie()));
    }
  }
}