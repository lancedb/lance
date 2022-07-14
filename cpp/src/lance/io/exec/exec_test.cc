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

#include "exec.h"

#include <arrow/dataset/dataset.h>
#include <arrow/dataset/scanner.h>
#include <arrow/table.h>
#include <arrow/type.h>

#include <memory>

#include "catch2/catch_test_macros.hpp"

auto DatasetSchema() {
  return ::arrow::schema({
      ::arrow::field("pk", ::arrow::utf8()),
      ::arrow::field("label", ::arrow::utf8()),
      ::arrow::field("value", ::arrow::int32()),
  });
}

auto CreateDataset() {
  auto table = ::arrow::Table::MakeEmpty(DatasetSchema()).ValueOrDie();
  return std::make_shared<::arrow::dataset::InMemoryDataset>(table);
}

TEST_CASE("SELECT * FROM dataset") {
  auto builder = ::arrow::dataset::ScannerBuilder(CreateDataset());
  auto scanner = builder.Finish().ValueOrDie();

  auto plan = lance::io::exec::Make(scanner->options()).ValueOrDie();
  INFO(plan->ToString());
}

TEST_CASE("SELECT pk WHERE label = 'car'") {}