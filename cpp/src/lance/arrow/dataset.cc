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

#include <algorithm>
#include <filesystem>
#include <random>

#include "lance/format/manifest.h"
#include "lance/format/schema.h"
#include "lance/io/writer.h"

namespace fs = std::filesystem;

namespace lance::arrow {

const std::string kLatestManifest = "_latest.manifest";
const std::string kDataDir = "data";

std::string GetManifestPath(const std::string& base_uri, std::optional<uint64_t> version) {
  if (version.has_value()) {
    return fs::path(base_uri) / "_versions" / fmt::format("{}.manifest", version.value());
  } else {
    return fs::path(base_uri) / kLatestManifest;
  }
}

class LanceDataset::Impl {
 public:
  Impl() = delete;

  Impl(std::shared_ptr<::arrow::fs::FileSystem> fs,
       const std::string& base_uri,
       std::shared_ptr<lance::format::Manifest> manifest)
      : fs_(std::move(fs)), base_uri_(base_uri), manifest_(std::move(manifest)) {}

  const std::shared_ptr<lance::format::Manifest>& manifest() const { return manifest_; }

 private:
  std::shared_ptr<::arrow::fs::FileSystem> fs_;
  std::string base_uri_;
  std::shared_ptr<lance::format::Manifest> manifest_;
};

LanceDataset::LanceDataset(std::unique_ptr<LanceDataset::Impl> impl)
    : ::arrow::dataset::Dataset(impl->manifest()->schema().ToArrow()), impl_(std::move(impl)) {}

LanceDataset::LanceDataset(const LanceDataset& other)
    : LanceDataset(std::make_unique<Impl>(*other.impl_)) {}

LanceDataset::~LanceDataset() {}

::arrow::Status LanceDataset::Write(const ::arrow::dataset::FileSystemDatasetWriteOptions& options,
                                    std::shared_ptr<::arrow::dataset::Scanner> scanner) {
  const auto& base_dir = options.base_dir;
  // Load previous latest Manifest if any.
  ARROW_ASSIGN_OR_RAISE(auto cur_dataset, LanceDataset::Make(options.filesystem, base_dir));

  std::shared_ptr<lance::format::Manifest> manifest;
  if (!cur_dataset) {
    // This is a completely new dataset, create Manifest with version 1.
    auto schema = std::make_shared<lance::format::Schema>(scanner->options()->dataset_schema);
    manifest = std::make_shared<lance::format::Manifest>(schema);
  } else {
    // Bump the version
    auto cur_manifest = cur_dataset->impl_->manifest();
    manifest = cur_manifest->BumpVersion();
  }
  // Write manifest file
  auto& fs = options.filesystem;

  auto lance_option = options;
  lance_option.base_dir = fs::path(base_dir) / kDataDir;
  auto partitioning = std::move(lance_option.partitioning);
  // TODO: support partition via lance manifest.
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
    return ::arrow::Status::OK();
  };
  lance_option.writer_post_finish = metadata_collector;
  // TODO: collecting files;

  ARROW_RETURN_NOT_OK(::arrow::dataset::FileSystemDataset::Write(lance_option, std::move(scanner)));

  // Write the manifest version file.
  // It only supports single writer at the moment.
  auto version_dir = (fs::path(base_dir) / "_versions").string();
  ARROW_RETURN_NOT_OK(fs->CreateDir(version_dir));
  auto manifest_path = GetManifestPath(base_dir, manifest->version());
  {
    ARROW_ASSIGN_OR_RAISE(auto out, fs->OpenOutputStream(manifest_path));
    ARROW_RETURN_NOT_OK(manifest->Write(out));
  }
  auto latest_manifest_path = GetManifestPath(base_dir, std::nullopt);
  return fs->CopyFile(manifest_path, latest_manifest_path);
}

::arrow::Result<std::shared_ptr<LanceDataset>> LanceDataset::Make(
    std::shared_ptr<::arrow::fs::FileSystem> fs,
    std::string base_uri,
    std::optional<uint64_t> version) {
  ARROW_ASSIGN_OR_RAISE(auto finfo, fs->GetFileInfo(base_uri));
  if (finfo.type() == ::arrow::fs::FileType::NotFound) {
    return nullptr;
  }
  auto manifest_path = GetManifestPath(base_uri, version);
  ARROW_ASSIGN_OR_RAISE(finfo, fs->GetFileInfo(base_uri));
  if (finfo.type() == ::arrow::fs::FileType::NotFound) {
    return ::arrow::Status::IOError("Manifest not found: ", manifest_path);
  }
  std::shared_ptr<lance::format::Manifest> manifest;
  {
    ARROW_ASSIGN_OR_RAISE(auto in, fs->OpenInputFile(manifest_path));
    ARROW_ASSIGN_OR_RAISE(auto parsed, lance::format::Manifest::Parse(in, 0));
    manifest = std::move(parsed);
  }
  auto impl = std::make_unique<LanceDataset::Impl>(fs, base_uri, manifest);
  return std::shared_ptr<LanceDataset>(new LanceDataset(std::move(impl)));
}

::arrow::Result<std::shared_ptr<::arrow::dataset::Dataset>> LanceDataset::ReplaceSchema(
    [[maybe_unused]] std::shared_ptr<::arrow::Schema> schema) const {
  return std::make_shared<LanceDataset>(*this);
}

::arrow::Result<::arrow::dataset::FragmentIterator> LanceDataset::GetFragmentsImpl(
    [[maybe_unused]] ::arrow::compute::Expression predicate) {
  std::vector<std::shared_ptr<::arrow::dataset::Fragment>> fragments;
  fragments.insert(fragments.begin(),
                   impl_->manifest()->fragments().begin(),
                   impl_->manifest()->fragments().end());
  return ::arrow::MakeVectorIterator(fragments);
}

}  // namespace lance::arrow