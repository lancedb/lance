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

#include "lance/arrow/dataset.h"

#include <arrow/dataset/api.h>
#include <arrow/status.h>
#include <fmt/format.h>
#include <uuid.h>

#include <filesystem>
#include <random>

#include "lance/format/manifest.h"
#include "lance/format/schema.h"
#include "lance/io/writer.h"

namespace fs = std::filesystem;

namespace lance::arrow {

::arrow::Status LanceDataset::Write(const ::arrow::dataset::FileSystemDatasetWriteOptions& options,
                                    std::shared_ptr<::arrow::dataset::Scanner> scanner) {
  // Write manifest file
  auto schema = std::make_shared<lance::format::Schema>(scanner->options()->dataset_schema);
  auto manifest = lance::format::Manifest(schema);
  auto& fs = options.filesystem;

  auto lance_option = options;
  lance_option.base_dir += "/data";
  auto partitioning = std::move(lance_option.partitioning);
  // TODO: support partition via lance manifest .
  lance_option.partitioning =
      std::make_shared<::arrow::dataset::FilenamePartitioning>(::arrow::schema({}));

  std::random_device rd;
  auto seed_data = std::array<int, std::mt19937::state_size>{};
  std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
  std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
  std::mt19937 generator(seq);
  uuids::uuid_random_generator gen{generator};
  auto uuid = gen();
  lance_option.basename_template = uuids::to_string(uuid) + "_{i}.lance";

  if (lance_option.format() == nullptr || lance_option.format()->type_name() != "lance") {
    return ::arrow::Status::Invalid("Must write with Lance format");
  }

  auto metadata_collector = [](::arrow::dataset::FileWriter* writer) {
    auto w = dynamic_cast<lance::io::FileWriter*>(writer);
    assert(w != nullptr);
    fmt::print("Writer: {}\n", writer->destination().path);
    return ::arrow::Status::OK();
  };
  lance_option.writer_post_finish = metadata_collector;

  ARROW_RETURN_NOT_OK(::arrow::dataset::FileSystemDataset::Write(lance_option, std::move(scanner)));

  // Write manifest file
  auto version_dir = (fs::path(options.base_dir) / "_versions").string();
  ARROW_RETURN_NOT_OK(fs->CreateDir(version_dir));
  auto manifest_path =
      (fs::path(version_dir) / fmt::format("{}.manifest", manifest.version())).string();
  {
    ARROW_ASSIGN_OR_RAISE(auto out, fs->OpenOutputStream(manifest_path));
    ARROW_RETURN_NOT_OK(manifest.Write(out));
  }
  auto latest_manifest_path = (fs::path(options.base_dir) / "_latest.manifest").string();
  return fs->CopyFile(manifest_path, latest_manifest_path);
}

}  // namespace lance::arrow