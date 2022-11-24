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
#include <fmt/chrono.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <argparse/argparse.hpp>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "bench_utils.h"
#include "lance/arrow/dataset.h"
#include "lance/arrow/scanner.h"

// we don't use catch2 Benchmark, as it adds heavy noise during profiling.
void BenchmarkFilterWithLimit(const std::string& uri, int samples = 5) {
  auto dataset = OpenDataset(uri);
  std::vector<std::chrono::milliseconds> durations;

  for (int i = 0; i < samples; i++) {
    auto start = std::chrono::system_clock::now();
    auto builder = lance::arrow::ScannerBuilder(dataset);
    assert(builder.Project({"image", "class"}).ok());
    assert(builder
               .Filter(::arrow::compute::equal(::arrow::compute::field_ref("class"),
                                               ::arrow::compute::literal("pug")))
               .ok());
    assert(builder.Limit(20, 50).ok());
    auto scanner = builder.Finish().ValueOrDie();
    auto table = scanner->ToTable().ValueOrDie();
    durations.emplace_back(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start));
  }
  fmt::print("Times: {}\n", durations);
}

int main(int argc, char** argv) {
  argparse::ArgumentParser parser(argv[0]);
  parser.add_argument("uri").help("Dataset URI");

  try {
    parser.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << parser;
    std::exit(1);
  }

  BenchmarkFilterWithLimit(parser.get("uri"));
  return 0;
}