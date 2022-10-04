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

#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "lance/arrow/stl.h"
#include "lance/arrow/type.h"
#include "lance/arrow/utils.h"
#include "lance/format/schema.h"
#include "lance/testing/extension_types.h"
#include "lance/testing/io.h"
#include "lance/testing/json.h"

using lance::arrow::ToArray;
using lance::testing::ArrayFromJSON;
using lance::testing::TableFromJSON;

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
  CHECK(scanner_builder.Limit(10).ok());
  CHECK(scanner_builder.Project({"objects.val"}).ok());
  CHECK(scanner_builder
            .Filter(::arrow::compute::equal(::arrow::compute::field_ref({"objects", 0, "val"}),
                                            ::arrow::compute::literal(2)))
            .ok());
  auto result = scanner_builder.Finish();
  CHECK(result.ok());
  auto scanner = result.ValueOrDie();

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

  auto schema =
      ::arrow::schema({arrow::field("c1", ::arrow::utf8()), arrow::field("c2", ext_type)});
  std::vector<std::shared_ptr<::arrow::Array>> cols;
  cols.push_back(c1);
  cols.push_back(::arrow::ExtensionType::WrapArray(ext_type, c2));
  return ::arrow::Table::Make(std::move(schema), std::move(cols));
}

std::shared_ptr<::arrow::dataset::Scanner> MakeScanner(std::shared_ptr<::arrow::Table> table) {
  auto dataset = std::make_shared<::arrow::dataset::InMemoryDataset>(table);
  auto scanner_builder = lance::arrow::ScannerBuilder(dataset);
  CHECK(scanner_builder.Limit(2).ok());
  CHECK(scanner_builder.Project({"c2"}).ok());
  // TODO how can extension types implement comparisons for filtering against storage type?
  auto result = scanner_builder.Finish();
  CHECK(result.ok());
  auto scanner = result.ValueOrDie();
  return scanner;
}

TEST_CASE("Scanner with extension") {
  auto ext_type = std::make_shared<::lance::testing::ParametricType>(1);
  CHECK(::arrow::RegisterExtensionType(ext_type).ok());
  auto table = MakeTable();
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

::arrow::Result<std::shared_ptr<::arrow::dataset::Scanner>> MakeScannerForBatchScan(
    int64_t num_values, int64_t batch_size) {
  std::vector<int32_t> values(num_values);
  std::iota(values.begin(), values.end(), 0);
  auto arr = lance::arrow::ToArray(values).ValueOrDie();
  auto table =
      ::arrow::Table::Make(::arrow::schema({::arrow::field("value", ::arrow::int32())}), {arr});

  auto dataset = std::make_shared<::arrow::dataset::InMemoryDataset>(table);
  auto scanner_builder = lance::arrow::ScannerBuilder(dataset);
  CHECK(scanner_builder.BatchSize(batch_size).ok());
  return scanner_builder.Finish();
}

TEST_CASE("Test Scanner::ToRecordBatchReader with batch size") {
  const int kTotalValues = 100;
  const int kBatchSize = 4;
  auto scanner = MakeScannerForBatchScan(kTotalValues, kBatchSize).ValueOrDie();
  auto record_batch_reader = scanner->ToRecordBatchReader().ValueOrDie();
  int num_batches = 0;
  while (auto batch = record_batch_reader->Next().ValueOrDie()) {
    CHECK(batch->num_rows() == kBatchSize);
    num_batches++;
  }
  CHECK(num_batches == kTotalValues / kBatchSize);
}

TEST_CASE("Test Scanner::ScanBatch with batch size") {
  const int kTotalValues = 100;
  const int kBatchSize = 4;
  auto scanner = MakeScannerForBatchScan(kTotalValues, kBatchSize).ValueOrDie();
  auto batches = scanner->ScanBatches().ValueOrDie();
  int num_batches = 0;
  while (true) {
    auto batch = batches.Next().ValueOrDie();
    if (!batch.record_batch) {
      break;
    }
    CHECK(batch.record_batch->num_rows() == kBatchSize);
    num_batches++;
  }
  CHECK(num_batches == kTotalValues / kBatchSize);
}

TEST_CASE("Test ScanBatchesAsync with batch size") {
  const int kTotalValues = 100;
  const int kBatchSize = 4;
  auto scanner = MakeScannerForBatchScan(kTotalValues, kBatchSize).ValueOrDie();
  auto generator = scanner->ScanBatchesAsync().ValueOrDie();
  int num_batches = 0;
  while (true) {
    auto fut = generator();
    CHECK(fut.Wait(1));
    auto batch = fut.result().ValueOrDie();
    if (!batch.record_batch) {
      break;
    }
    num_batches++;
    CHECK(batch.record_batch->num_rows() == kBatchSize);
  }
  CHECK(num_batches == kTotalValues / kBatchSize);
}

// GH-188
TEST_CASE("Filter over empty list") {
  auto schema = ::arrow::schema({::arrow::field("ints", ::arrow::int32()),
                                 ::arrow::field("floats", ::arrow::list(::arrow::float32()))});
  auto t = TableFromJSON(schema, R"([
{"ints": 1, "floats": [0.1, 0.2]},
{"ints": 2},
{"ints": 3, "floats": [11.1]}
])").ValueOrDie();

  auto dataset = lance::testing::MakeDataset(t).ValueOrDie();
  auto scan_builder = dataset->NewScan().ValueOrDie();

  // This filter should result in an empty list array
  CHECK(scan_builder
            ->Filter(::arrow::compute::equal(::arrow::compute::field_ref("ints"),
                                             ::arrow::compute::literal(100)))
            .ok());
  auto scanner = scan_builder->Finish().ValueOrDie();

  auto actual = scanner->ToTable().ValueOrDie();
  CHECK(actual->num_rows() == 0);
  CHECK(t->schema()->Equals(actual->schema()));
}

TEST_CASE("Filter with limit") {
  auto schema = ::arrow::schema({::arrow::field("ints", ::arrow::int32()),
                                 ::arrow::field("floats", ::arrow::list(::arrow::float32()))});
  auto t = TableFromJSON(schema, R"([
{"ints": 1, "floats": [0.1, 0.2]},
{"ints": 2, "floats": [11.1]}
])").ValueOrDie();

  auto dataset = lance::testing::MakeDataset(t).ValueOrDie();
  auto scan_builder = lance::arrow::ScannerBuilder(dataset);
  CHECK(scan_builder
            .Filter(::arrow::compute::equal(::arrow::compute::field_ref("ints"),
                                            ::arrow::compute::literal(100)))
            .ok());
  CHECK(scan_builder.Limit(20).ok());

  auto scanner = scan_builder.Finish().ValueOrDie();
  auto actual = scanner->ToTable().ValueOrDie();
  CHECK(actual->num_rows() == 0);
  CHECK(t->schema()->Equals(actual->schema()));
}

TEST_CASE("Scanner projection should not include filter columns") {
  auto schema = ::arrow::schema(
      {::arrow::field("ints", ::arrow::int32(), false), ::arrow::field("strs", ::arrow::utf8())});
  auto t = TableFromJSON(schema, R"([{"ints": 1, "strs": "one"}])").ValueOrDie();
  auto dataset = lance::testing::MakeDataset(t).ValueOrDie();
  auto scan_builder = lance::arrow::ScannerBuilder(dataset);
  CHECK(scan_builder
            .Filter(::arrow::compute::equal(::arrow::compute::field_ref("ints"),
                                            ::arrow::compute::literal(2)))
            .ok());
  CHECK(scan_builder.Project({"strs"}).ok());

  auto scanner = scan_builder.Finish().ValueOrDie();
  auto actual = scanner->ToTable().ValueOrDie();
  auto expected_schema = ::arrow::schema({::arrow::field("strs", ::arrow::utf8())});
  INFO("Expected schema: " << expected_schema->ToString()
                           << "\nGot: " << actual->schema()->ToString());
  CHECK(actual->schema()->Equals(expected_schema));
}

TEST_CASE("Test filter with smaller batch size than block size") {
  std::vector<int32_t> ints(200);
  std::iota(std::begin(ints), std::end(ints), 0);
  auto ints_arr = ToArray(ints).ValueOrDie();
  std::vector<std::string> strs(ints.size());
  std::transform(
      std::begin(ints), std::end(ints), std::begin(strs), [](auto v) { return std::to_string(v); });
  auto strs_arr = ToArray(strs).ValueOrDie();

  auto schema = ::arrow::schema(
      {::arrow::field("ints", ::arrow::int32()), ::arrow::field("strs", ::arrow::utf8())});
  auto table = ::arrow::Table::Make(schema, {ints_arr, strs_arr});

  const uint64_t kGroupSize = 64;
  auto dataset = lance::testing::MakeDataset(table, {}, kGroupSize).ValueOrDie();

  auto scan_builder = lance::arrow::ScannerBuilder(dataset);
  // WHERE ints % 5 == 0
  auto status = scan_builder.Filter(::arrow::compute::equal(
      ::arrow::compute::call(
          "subtract",
          {::arrow::compute::field_ref("ints"),
           ::arrow::compute::call(
               "multiply",
               {::arrow::compute::call(
                    "divide", {::arrow::compute::field_ref("ints"), ::arrow::compute::literal(5)}),
                ::arrow::compute::literal(5)})}),
      ::arrow::compute::literal(0)));
  INFO("Build filter status: " << status.message());
  CHECK(status.ok());
  CHECK(scan_builder.Project({"strs"}).ok());
  CHECK(scan_builder.BatchSize(7).ok());  // Some number that is not dividable by the group size.
  auto scanner = scan_builder.Finish().ValueOrDie();
  auto actual = scanner->ToTable().ValueOrDie();

  std::vector<std::string> expected_strs;
  for (size_t i = 0; i < ints.size() / 5; i++) {
    expected_strs.emplace_back(std::to_string(i * 5));
  }
  auto expected_arr = ToArray(expected_strs).ValueOrDie();
  auto expected = ::arrow::Table::Make(::arrow::schema({::arrow::field("strs", ::arrow::utf8())}),
                                       {expected_arr});
  CHECK(actual->Equals(*expected));
}

// GH-204
TEST_CASE("Test projection over nested field") {
  auto schema =
      ::arrow::schema({::arrow::field("id", ::arrow::int64()),
                       ::arrow::field("annotations",
                                      ::arrow::struct_({
                                          ::arrow::field("name", ::arrow::list(::arrow::utf8())),
                                          ::arrow::field("value", ::arrow::list(::arrow::int32())),
                                      }))});
  auto table =
      TableFromJSON(schema, R"([{"id": 1, "annotations": {"name": ["a", "b"], "value": [1]}}])")
          .ValueOrDie();
  fmt::print("Table is: {}\n", table->ToString());

  auto dataset = lance::testing::MakeDataset(table).ValueOrDie();
  auto scan_builder = lance::arrow::ScannerBuilder(dataset);
  CHECK(scan_builder.Project({"annotations.name"}).ok());
  auto scanner = scan_builder.Finish().ValueOrDie();
  auto result = scanner->ToTable();
  INFO("Scanner to table result: " << result.status().ToString());
  CHECK(result.ok());
  auto actual = result.ValueOrDie();
  auto expected_schema = ::arrow::schema({::arrow::field(
      "annotations", ::arrow::struct_({::arrow::field("name", ::arrow::list(::arrow::utf8()))}))});
  CHECK(actual->schema()->Equals(expected_schema));
  auto expected_table =
      TableFromJSON(expected_schema, R"([{"annotations": {"name": ["a", "b"]}}])").ValueOrDie();
  CHECK(actual->Equals(*expected_table));
}