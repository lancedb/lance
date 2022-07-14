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

#include "lance/arrow/type.h"

#include <arrow/type.h>
#include <fmt/format.h>
#include <google/protobuf/util/message_differencer.h>

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <string>
#include <vector>

#include "lance/format/format.pb.h"

using google::protobuf::util::MessageDifferencer;
using std::string;
using std::vector;

namespace pb = lance::format::pb;
using lance::arrow::FromArrowSchema;

TEST_CASE("To arrow schema") {
  // COCO dataset
  auto annotationType = arrow::struct_(
      {arrow::field("label", arrow::utf8()), arrow::field("score", arrow::float32())});
  auto schema =
      arrow::schema({arrow::field("filename", arrow::utf8()),
                     arrow::field("split", arrow::utf8()),
                     arrow::field("width", arrow::int32()),
                     arrow::field("image", arrow::struct_({arrow::field("uri", arrow::utf8())})),
                     arrow::field("annotations", arrow::list(annotationType))});

  auto result = FromArrowSchema(schema);
  CHECK(result.ok());
  CHECK(result->size() == 9);

  auto actual_schema = lance::arrow::ToArrowSchema(result.ValueOrDie());
  CHECK(actual_schema.ok());
  INFO("Expect schema: " << schema->ToString() << "\n, Actual: " << (*actual_schema)->ToString());
  CHECK((*actual_schema)->Equals(schema));
};

TEST_CASE("Arrow Schema (simple)") {
  auto schema =
      arrow::schema({arrow::field("pk", arrow::utf8()), arrow::field("value", arrow::int64())});

  auto fields = lance::arrow::FromArrowSchema(schema).ValueOrDie();
  CHECK(fields.size() == 2);

  pb::Field expected;
  expected.set_id(0);
  expected.set_type(pb::Field::LEAF);
  expected.set_logical_type("string");
  expected.set_name("pk");
  expected.set_parent_id(-1);
  expected.set_dictionary_offset(-1);
  expected.set_encoding(pb::Encoding::VAR_BINARY);

  CHECK(MessageDifferencer::Equals(expected, fields.at(0)));

  expected.set_id(1);
  expected.set_type(pb::Field::LEAF);
  expected.set_logical_type("int64");
  expected.set_name("value");
  expected.set_parent_id(-1);
  expected.set_dictionary_offset(-1);
  expected.set_encoding(pb::Encoding::PLAIN);

  INFO("Actual fields: " << fields[1].DebugString() << " expect: " << expected.DebugString());
  CHECK(MessageDifferencer::Equals(expected, fields.at(1)));

  auto actual_schema = lance::arrow::ToArrowSchema(fields).ValueOrDie();
  CHECK(actual_schema->Equals(schema));
}

TEST_CASE("List schema") {
  auto schema = arrow::schema({arrow::field("ints", arrow::list(arrow::int32()))});

  //  auto fields = lance::arrow::FromArrowSchema(schema).ValueOrDie();
  auto result = lance::arrow::FromArrowSchema(schema);
  INFO("Convert to fields: " << result.status());
  CHECK(result.ok());

  auto actual_schema = lance::arrow::ToArrowSchema(*result);
  CHECK(result.ok());
  CHECK((*actual_schema)->Equals(schema));
}

TEST_CASE("Parse dictionary type") {
  auto dict_type = arrow::dictionary(arrow::uint16(), arrow::utf8(), false);
  auto logical_type = lance::arrow::ToLogicalType(dict_type).ValueOrDie();
  CHECK(logical_type == "dict:string:uint16:false");

  auto actual = lance::arrow::FromLogicalType(logical_type).ValueOrDie();
  CHECK(dict_type->Equals(actual));

  dict_type = arrow::dictionary(arrow::int32(), arrow::utf8(), true);
  logical_type = lance::arrow::ToLogicalType(dict_type).ValueOrDie();
  CHECK(logical_type == "dict:string:int32:true");

  actual = lance::arrow::FromLogicalType(logical_type).ValueOrDie();
  CHECK(dict_type->Equals(actual));
}