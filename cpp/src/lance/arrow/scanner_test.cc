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

#include "lance/arrow/scanner.h"

#include <arrow/builder.h>
#include <arrow/dataset/dataset.h>
#include <arrow/dataset/scanner.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "lance/arrow/type.h"
#include "lance/arrow/testing.h"
#include "lance/format/schema.h"


auto nested_schema = ::arrow::schema({::arrow::field("pk", ::arrow::int32()),
                                      ::arrow::field("objects",
                                                     ::arrow::list(::arrow::struct_({
                                                         ::arrow::field("val", ::arrow::int64()),
                                                         ::arrow::field("id", ::arrow::int32()),
                                                         ::arrow::field("label", ::arrow::utf8()),
                                                     })))});

TEST_CASE("Project nested columns") {
  auto schema = ::arrow::schema({::arrow::field("objects",
                                                ::arrow::list(::arrow::struct_({
                                                    ::arrow::field("val", ::arrow::int64()),
                                                })))});
  auto fields = schema->GetFieldByName("objects");
  fmt::print("Fields: {} {}\n", fields, fields->type()->field(0)->type()->field(0));

  auto ref = ::arrow::FieldRef("objects", 0, "val");
  auto f = ref.FindOne(*schema).ValueOrDie();
  fmt::print("FindAll: {}\n", f.ToString());
  CHECK(!f.empty());

  auto expr = ::arrow::compute::field_ref({"objects", 0, "val"});
  fmt::print("Expr field: {} {}\n", expr.field_ref()->ToString(), expr.field_ref()->ToDotPath());
  f = expr.field_ref()->FindOne(*schema).ValueOrDie();
  fmt::print("FindAll: {}\n", f.ToString());
  CHECK(!f.empty());
}

TEST_CASE("Build Scanner with nested struct") {
  auto table = ::arrow::Table::MakeEmpty(nested_schema).ValueOrDie();
  auto dataset = std::make_shared<::arrow::dataset::InMemoryDataset>(table);
  auto scanner_builder = lance::arrow::ScannerBuilder(dataset);
  scanner_builder.Limit(10);
  scanner_builder.Project({"objects.val"});
  scanner_builder.Filter(::arrow::compute::equal(::arrow::compute::field_ref({"objects", 0, "val"}),
                                                 ::arrow::compute::literal(2)));
  auto result = scanner_builder.Finish();
  CHECK(result.ok());
  auto scanner = result.ValueOrDie();
  fmt::print("Projected: {}\n", scanner->options()->projected_schema);

  auto expected_proj_schema = ::arrow::schema({::arrow::field(
      "objects", ::arrow::list(::arrow::struct_({::arrow::field("val", ::arrow::int64())})))});
  INFO("Expected schema: " << expected_proj_schema->ToString());
  INFO("Actual schema: " << scanner->options()->projected_schema->ToString());
  CHECK(expected_proj_schema->Equals(scanner->options()->projected_schema));

  CHECK(scanner->options()->batch_size == 10);
  CHECK(scanner->options()->batch_readahead == 1);

  fmt::print("Scanner Options: {}\n", scanner->options()->filter.ToString());
}


std::shared_ptr<::arrow::Table> MakeTable() {
  auto ext_type = std::make_shared<::lance::testing::ParametricType>(1);
  ::arrow::StringBuilder stringBuilder;
  ::arrow::Int32Builder intBuilder;

  CHECK(stringBuilder.AppendValues({"train", "train", "split", "train"}).ok());
  auto c1 = stringBuilder.Finish().ValueOrDie();
  stringBuilder.Reset();

  CHECK(intBuilder.AppendValues({1, 2, 3, 4}).ok());
  auto c2 = intBuilder.Finish().ValueOrDie();
  intBuilder.Reset();

  auto schema = ::arrow::schema({arrow::field("c1", ::arrow::utf8()),
                                 arrow::field("c2", ext_type)});

  std::vector<std::shared_ptr<::arrow::Array>> cols;
  cols.push_back(c1);
  cols.push_back(::arrow::ExtensionType::WrapArray(ext_type, c2));
  return ::arrow::Table::Make(std::move(schema), std::move(cols));
}

std::shared_ptr<::arrow::dataset::Scanner> MakeScanner(std::shared_ptr<::arrow::Table> table) {
  auto dataset = std::make_shared<::arrow::dataset::InMemoryDataset>(table);
  auto scanner_builder = lance::arrow::ScannerBuilder(dataset);
  scanner_builder.Limit(2);
  scanner_builder.Project({"c2"});
  // TODO how can extension types implement comparisons for filtering against storage type?
  auto result = scanner_builder.Finish();
  CHECK(result.ok());
  auto scanner = result.ValueOrDie();
  fmt::print("Projected: {}\n", scanner->options()->projected_schema);
  return scanner;
}


TEST_CASE("Scanner with extension") {
  auto table = MakeTable();
  auto ext_type = std::make_shared<::lance::testing::ParametricType>(1);
  ::arrow::RegisterExtensionType(ext_type);
  auto scanner = MakeScanner(table);

  auto dataset = std::make_shared<::arrow::dataset::InMemoryDataset>(table);
  INFO("Dataset schema is " << dataset->schema()->ToString());

  auto schema = ::lance::format::Schema(dataset->schema());
  INFO("Lance schema is " << schema.ToString());

  auto expected_proj_schema = ::arrow::schema({::arrow::field("c2", ext_type)});
  INFO("Expected schema: " << expected_proj_schema->ToString());
  INFO("Actual schema: " << scanner->options()->projected_schema->ToString());
  CHECK(expected_proj_schema->Equals(scanner->options()->projected_schema));

  auto actual_table = scanner->ToTable().ValueOrDie();
  CHECK(actual_table->schema()->Equals(expected_proj_schema));
  CHECK(actual_table->GetColumnByName("c2")->type()->Equals(ext_type));
}