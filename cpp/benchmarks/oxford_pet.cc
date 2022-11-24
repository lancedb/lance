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

#include <arrow/compute/api.h>
#include <arrow/filesystem/api.h>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "bench_utils.h"
#include "lance/arrow/dataset.h"
#include "lance/arrow/scanner.h"

std::string uri;

TEST_CASE("SELECT image, class FROM ds WHERE class='pug' LIMIT 50 OFFSET 20") {
  CHECK(!uri.empty());
  auto fs = std::make_shared<arrow::fs::LocalFileSystem>();

  auto result = lance::arrow::LanceDataset::Make(fs, uri);
  INFO(result.status().message());
  auto dataset = OpenDataset(uri);
  BENCHMARK("Run query") {
    auto builder = lance::arrow::ScannerBuilder(dataset);
    CHECK(builder.Project({"image", "class"}).ok());
    CHECK(builder
              .Filter(::arrow::compute::equal(::arrow::compute::field_ref("class"),
                                              ::arrow::compute::literal("pug")))
              .ok());
    CHECK(builder.Limit(20, 50).ok());
    auto scanner = builder.Finish().ValueOrDie();
    auto table = scanner->ToTable().ValueOrDie();
  };
}

int main(int argc, char** argv) {
  Catch::Session session;
  using namespace Catch::Clara;

  auto cli = session.cli() | Opt(uri, "uri")["--uri"]("Input file URI");
  session.cli(cli);

  int ret = session.applyCommandLine(argc, argv);
  if (ret != 0) {
    return ret;
  }
  return session.run();
}