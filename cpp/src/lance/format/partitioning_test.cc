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

#include "lance/format/partitioning.h"

#include <arrow/type.h>

#include <algorithm>
#include <catch2/catch_test_macros.hpp>

#include "lance/format/schema.h"

using namespace ::arrow::compute;

TEST_CASE("Create partition from protobuf") {
  auto arrow_schema = ::arrow::schema(
      {::arrow::field("a", ::arrow::int32()), ::arrow::field("split", ::arrow::utf8())});
  auto schema = std::make_shared<::lance::format::Schema>(arrow_schema);

  auto partitioning = lance::format::Partitioning::Make(std::move(schema)).ValueOrDie();

  auto arrow_part = partitioning.ToArrow();
  CHECK(arrow_part->type_name() == "hive");
  auto path = arrow_part
                  ->Format(::arrow::compute::and_(
                      ::arrow::compute::equal(::arrow::compute::field_ref("a"),
                                              ::arrow::compute::literal(123)),
                      ::arrow::compute::equal(::arrow::compute::field_ref("split"),
                                              ::arrow::compute::literal("test"))))
                  .ValueOrDie();
  CHECK(path.directory == "a=123/split=test");

  auto expr = arrow_part->Parse("a=12/split=eval/foo.lance").ValueOrDie();
  CHECK(expr.Equals(
      and_(equal(field_ref("a"), literal(12)), equal(field_ref("split"), literal("eval")))));

  auto proto = partitioning.ToProto();

  auto field_ids = partitioning.schema()->GetFieldIds();
  std::equal(proto.field_ids().begin(), proto.field_ids().end(), field_ids.begin());
  CHECK(proto.field_ids(0) == 0);
}