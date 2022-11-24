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

#include <arrow/filesystem/filesystem.h>
#include <parquet/arrow/reader.h>

#include <argparse/argparse.hpp>
#include <iostream>
#include <map>

#include "lance/arrow/writer.h"
#include "bench_utils.h"

using std::map;
using std::string;

namespace fs = std::filesystem;

/// Convert a Parquet file to lance file.
arrow::Status ConvertParquet(const std::string& in_uri, const std::string& out) {
  auto uri = in_uri;
  std::string path;
  auto fs = ::arrow::fs::FileSystemFromUriOrPath(uri, &path).ValueOrDie();

  auto dataset = OpenDataset(in_uri, "parquet");
  auto scan_builder = dataset->NewScan().ValueOrDie();
  auto scanner = scan_builder->Finish().ValueOrDie();
  auto table = scanner->ToTable().ValueOrDie();

  auto outfile = fs->OpenOutputStream(out).ValueOrDie();
  return lance::arrow::WriteTable(*table, outfile);
}

int main(int argc, char** argv) {
  argparse::ArgumentParser parser("convert");
  parser.add_description("Convert parquet format to lance format");

  parser.add_argument("-i", "--input").help("File path to rikai parquet file");
  parser.add_argument("-o", "--output").help("output filepath");

  try {
    parser.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << parser;
    std::exit(1);
  }

  auto input_file = parser.get("input");
  auto output_file = parser.get("output");
  auto status = ConvertParquet(input_file, output_file);

  if (!status.ok()) {
    std::cerr << status << std::endl;
    std::exit(1);
  }

  return 0;
}