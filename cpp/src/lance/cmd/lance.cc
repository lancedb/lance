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

#include <arrow/dataset/api.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <argparse/argparse.hpp>
#include <iostream>
#include <string>

#include "lance/arrow/type.h"
#include "lance/arrow/utils.h"
#include "lance/io/reader.h"

using std::string;

void PrintLine(int width = 40) { fmt::print("{:-^{}}\n", "", width); }

::arrow::Status PrintSchema(const std::shared_ptr<::arrow::dataset::FileSystemDataset>& dataset) {
  auto files = dataset->files();
  auto infile = dataset->filesystem()->OpenInputFile(files[0]).ValueOrDie();
  auto reader = lance::io::FileReader::Make(infile).ValueOrDie();
  auto schema = reader->schema();
  PrintLine();
  fmt::print("Lance schema:\n", schema);
  lance::format::Print(schema);
  return ::arrow::Status::OK();
}

::arrow::Status PrintSummary(const std::shared_ptr<::arrow::dataset::FileSystemDataset>& dataset) {
  assert(dataset);
  PrintLine();
  fmt::print("Summary: \n");
  int num_batches = 0;
  int total = 0;
  for (auto file : dataset->files()) {
    ARROW_ASSIGN_OR_RAISE(auto infile, dataset->filesystem()->OpenInputFile(file));
    ARROW_ASSIGN_OR_RAISE(auto reader, lance::io::FileReader::Make(infile));
    num_batches += reader->num_batches();
    total += reader->length();
  }
  fmt::print("  Total records: {}\n", total);
  fmt::print("  Number of batches: {}\n", num_batches);
  fmt::print("  Mean batch size: {}\n", total / num_batches + 1);
  return ::arrow::Status::OK();
}

::arrow::Status inspect(const argparse::ArgumentParser& args) {
  auto uri = args.get<string>("uri");
  fmt::print("Inspecting dataset: {}\n", uri);
  ARROW_ASSIGN_OR_RAISE(auto dataset, lance::arrow::OpenDataset(uri));
  ARROW_RETURN_NOT_OK(PrintSummary(dataset));
  ARROW_RETURN_NOT_OK(PrintSchema(dataset));

  return ::arrow::Status::OK();
}

int main(int argc, char** argv) {
  argparse::ArgumentParser parser("lq");
  parser.add_description("lq: command-line lance inspector");

  argparse::ArgumentParser inspect_parser("inspect");
  inspect_parser.add_description("Inspect dataset");
  inspect_parser.add_argument("uri").help("Dataset URI").required();
  parser.add_subparser(inspect_parser);

  try {
    parser.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << "\n";
    std::cerr << parser;
    std::exit(1);
  }

  ::arrow::Status status;
  if (parser.is_subcommand_used("inspect")) {
    status = inspect(inspect_parser);
  }
  if (!status.ok()) {
    std::cerr << status.ToString() << "\n";
    std::exit(1);
  }
  return 0;
}
