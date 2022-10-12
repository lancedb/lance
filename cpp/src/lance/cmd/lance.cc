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

int inspect(const argparse::ArgumentParser& args) {
  auto uri = args.get<string>("uri");
  fmt::print("Inspecting dataset: {}\n", uri);
  auto result = lance::arrow::OpenDataset(uri);
  if (!result.ok()) {
    fmt::print(stderr, "{}\n", result.status());
    return -1;
  }
  auto dataset = result.ValueOrDie();
  auto frag_result = dataset->GetFragments();
  if (!frag_result.ok()) {
    fmt::print(stderr, "{}\n", frag_result.status().ToString());
    return -1;
  }
  auto fragments = std::move(frag_result.ValueOrDie());
  auto file_fragment =
      std::dynamic_pointer_cast<::arrow::dataset::FileFragment>(fragments.Next().ValueOrDie());
  auto infile = dataset->filesystem()->OpenInputFile(file_fragment->source().path()).ValueOrDie();
  auto reader = lance::io::FileReader::Make(infile).ValueOrDie();
  auto schema = reader->schema();
  fmt::print("Lance schema:\n{}\n", schema);

  return 0;
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

  if (parser.is_subcommand_used("inspect")) {
    return inspect(inspect_parser);
  }
  return 0;
}
