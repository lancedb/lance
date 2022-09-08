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

#include <arrow/dataset/api.h>
#include <fmt/format.h>

#include <argparse/argparse.hpp>
#include <cstdlib>
#include <string>
#include <vector>

#include "bench_utils.h"
#include "lance/arrow/scanner.h"

void Scan(const std::string& uri, const std::vector<std::string>& columns) {
  auto dataset = OpenDataset(uri);
  auto scan_builder = lance::arrow::ScannerBuilder(dataset);
  scan_builder.Project(columns);
  auto scanner = scan_builder.Finish().ValueOrDie();
  fmt::print("Dataset count: {}\n", scanner->CountRows().ValueOrDie());
}

int main(int argc, char* argv[]) {
  argparse::ArgumentParser parser("scan");
  parser.add_description("Profile scanning performance");

  parser.add_argument("uri").help("Dataset URI");

  try {
    parser.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << parser;
    std::exit(1);
  }

  Scan(parser.get("uri"), {"class"});
}
