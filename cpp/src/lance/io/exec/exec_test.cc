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

#include "lance/io/exec/exec.h"

#include <arrow/compute/exec/expression.h>
#include <arrow/dataset/dataset.h>
#include <arrow/dataset/scanner.h>
#include <arrow/table.h>
#include <arrow/type.h>

#include <memory>

#include "catch2/catch_test_macros.hpp"
#include "lance/arrow/scan_options.h"
#include "lance/format/schema.h"

auto TestSchema() {
  return ::arrow::schema({
      ::arrow::field("pk", ::arrow::utf8()),
      ::arrow::field("label", ::arrow::utf8()),
      ::arrow::field("value", ::arrow::int32()),
  });
}

auto TestDataset() {
  auto table = ::arrow::Table::MakeEmpty(TestSchema()).ValueOrDie();
  return std::make_shared<::arrow::dataset::InMemoryDataset>(table);
}

TEST_CASE("SELECT * FROM dataset") {
  auto builder = ::arrow::dataset::ScannerBuilder(TestDataset());
  auto scanner = builder.Finish().ValueOrDie();

  auto options = std::make_shared<lance::arrow::ScanOptions>(
      std::make_shared<lance::format::Schema>(scanner->options()->dataset_schema),
      scanner->options());

  auto plan = lance::io::exec::Make(options).ValueOrDie();
  INFO(plan->ToString());
  CHECK(plan->Validate().ok());
  CHECK(plan->type_name() == "project");
}

TEST_CASE("SELECT pk WHERE label = 'car'") {
  auto builder = ::arrow::dataset::ScannerBuilder(TestDataset());
  CHECK(builder.Project({"pk"}).ok());
  CHECK(builder
            .Filter(::arrow::compute::equal(::arrow::compute::field_ref("label"),
                                            ::arrow::compute::literal("car")))
            .ok());
  auto scanner = builder.Finish().ValueOrDie();

  auto options = std::make_shared<lance::arrow::ScanOptions>(
      std::make_shared<lance::format::Schema>(scanner->options()->dataset_schema),
      scanner->options());

  auto plan = lance::io::exec::Make(options).ValueOrDie();
  CHECK(plan->Validate().ok());
}