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
#include <fmt/ranges.h>
#include <uuid.h>

#include <algorithm>
#include <filesystem>
#include <mutex>
#include <random>
#include <range/v3/all.hpp>
#include <utility>

#include "lance/arrow/file_lance_ext.h"
#include "lance/format/manifest.h"
#include "lance/format/schema.h"
#include "lance/io/writer.h"

namespace fs = std::filesystem;
using namespace ranges;

namespace lance::arrow {

const std::string kLatestManifest = "_latest.manifest";
const std::string kDataDir = "data";
const std::string kVersionsDir = "_versions";

namespace {

/// Get Manifest file path for a version.
///
/// \param base_uri the base uri of the dataset
/// \param version optional version number. If not specified, returns the latest version.
/// \return string path to the manifest file.
std::string GetManifestPath(const std::string& base_uri,
                            std::optional<uint64_t> version = std::nullopt) {
  if (version.has_value()) {
    return fs::path(base_uri) / kVersionsDir / fmt::format("{}.manifest", version.value());
  } else {
    return fs::path(base_uri) / kLatestManifest;
  }
}

auto CreateFragments(const std::vector<std::string>& paths, const lance::format::Schema& schema) {
  auto field_ids = schema.GetFieldIds();
  return paths | views::transform([&field_ids](auto& path) {
           return std::make_shared<lance::format::DataFragment>(
               lance::format::DataFile(path, field_ids));
         }) |
         to<std::vector<std::shared_ptr<lance::format::DataFragment>>>;
}

std::string GetBasenameTemplate() {
  std::random_device rd;
  auto seed_data = std::array<int, std::mt19937::state_size>{};
  std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
  std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
  std::mt19937 generator(seq);
  uuids::uuid_random_generator gen{generator};
  auto uuid = gen();

  return uuids::to_string(uuid) + "_{i}.lance";
}

::arrow::Result<std::shared_ptr<lance::format::Manifest>> OpenManifest(
    const std::shared_ptr<::arrow::fs::FileSystem>& fs, const std::string& path) {
  ARROW_ASSIGN_OR_RAISE(auto in, fs->OpenInputFile(path));
  return lance::format::Manifest::Parse(in, 0);
}

}  // namespace

DatasetVersion::DatasetVersion(uint64_t version) : version_(version) {}

uint64_t DatasetVersion::version() const { return version_; }

class LanceDataset::Impl {
 public:
  Impl() = delete;

  Impl(std::shared_ptr<::arrow::fs::FileSystem> filesystem,
       std::string uri,
       std::shared_ptr<lance::format::Manifest> m)
      : fs(std::move(filesystem)), base_uri(std::move(uri)), manifest(std::move(m)) {}

  std::string data_dir() const { return fs::path(base_uri) / kDataDir; }

  std::string versions_dir() const { return fs::path(base_uri) / kVersionsDir; }

  std::shared_ptr<::arrow::fs::FileSystem> fs;
  std::string base_uri;
  std::shared_ptr<lance::format::Manifest> manifest;
};

LanceDataset::LanceDataset(std::unique_ptr<LanceDataset::Impl> impl)
    : ::arrow::dataset::Dataset(impl->manifest->schema().ToArrow()), impl_(std::move(impl)) {}

LanceDataset::LanceDataset(const LanceDataset& other)
    : LanceDataset(std::make_unique<Impl>(*other.impl_)) {}

LanceDataset::~LanceDataset() {}

::arrow::Status LanceDataset::Write(const ::arrow::dataset::FileSystemDatasetWriteOptions& options,
                                    std::shared_ptr<::arrow::dataset::Scanner> scanner,
                                    WriteMode mode) {
  const auto& base_dir = options.base_dir;
  const auto data_dir = (fs::path(base_dir) / kDataDir).string();
  auto& fs = options.filesystem;

  std::shared_ptr<lance::format::Manifest> manifest;
  if (mode == kCreate) {
    if (fs->GetFileInfo(base_dir)->type() != ::arrow::fs::FileType::NotFound) {
      return ::arrow::Status::AlreadyExists("Dataset ", base_dir, " already exists");
    }
  } else {
    // Append or Overwrite
    ARROW_ASSIGN_OR_RAISE(auto cur_dataset, LanceDataset::Make(options.filesystem, base_dir));
    if (!cur_dataset) {
      if (mode == kAppend) {
        return ::arrow::Status::IOError("Append to non-existed dataset: ", base_dir);
      }
    } else {
      auto existing_manifest = cur_dataset->impl_->manifest;
      auto existing_arrow_schema = existing_manifest->schema().ToArrow();

      if (!scanner->dataset()->schema()->Equals(existing_arrow_schema)) {
        return ::arrow::Status::IOError("Write dataset with different schema: ",
                                        scanner->dataset()->schema()->ToString(),
                                        " != ",
                                        existing_arrow_schema->ToString());
      }

      // Bump the version
      manifest = cur_dataset->impl_->manifest->BumpVersion(mode == kOverwrite);
    }
  }
  if (!manifest) {
    // This is a completely new dataset, create Manifest with version 1.
    auto schema = std::make_shared<lance::format::Schema>(scanner->options()->dataset_schema);
    manifest = std::make_shared<lance::format::Manifest>(schema);
  }

  // Write manifest file
  auto lance_option = options;
  lance_option.base_dir = data_dir;
  lance_option.existing_data_behavior = ::arrow::dataset::ExistingDataBehavior::kOverwriteOrIgnore;
  auto partitioning = std::move(lance_option.partitioning);
  // TODO: support partition via lance manifest.
  lance_option.partitioning =
      std::make_shared<::arrow::dataset::HivePartitioning>(::arrow::schema({}));

  lance_option.basename_template = GetBasenameTemplate();

  if (lance_option.format() == nullptr || lance_option.format()->type_name() != "lance") {
    return ::arrow::Status::Invalid("Must write with Lance format");
  }

  std::vector<std::string> paths;
  std::mutex mutex;
  auto metadata_collector = [&paths, data_dir, &mutex](::arrow::dataset::FileWriter* writer) {
    auto w = dynamic_cast<lance::io::FileWriter*>(writer);
    assert(w != nullptr);
    auto relative = fs::relative(w->destination().path, data_dir);
    {
      std::lock_guard guard(mutex);
      paths.emplace_back(relative);
    }
    return ::arrow::Status::OK();
  };
  lance_option.writer_post_finish = metadata_collector;

  ARROW_RETURN_NOT_OK(::arrow::dataset::FileSystemDataset::Write(lance_option, std::move(scanner)));

  manifest->AppendFragments(CreateFragments(paths, manifest->schema()));
  // Write the manifest version file.
  // It only supports single writer at the moment.
  auto version_dir = (fs::path(base_dir) / kVersionsDir).string();
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
    const std::shared_ptr<::arrow::fs::FileSystem>& fs,
    const std::string& base_uri,
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
  ARROW_ASSIGN_OR_RAISE(auto manifest, OpenManifest(fs, manifest_path));
  auto impl = std::make_unique<LanceDataset::Impl>(fs, base_uri, manifest);
  return std::shared_ptr<LanceDataset>(new LanceDataset(std::move(impl)));
}

::arrow::Result<std::vector<DatasetVersion>> LanceDataset::versions() const {
  std::vector<DatasetVersion> versions;
  ::arrow::fs::FileSelector selector;
  selector.base_dir = impl_->versions_dir();
  selector.allow_not_found = true;
  selector.recursive = false;

  ARROW_ASSIGN_OR_RAISE(auto file_infos, impl_->fs->GetFileInfo(selector));
  for (const auto& finfo : file_infos) {
    ARROW_ASSIGN_OR_RAISE(auto manifest, OpenManifest(impl_->fs, finfo.path()));
    versions.emplace_back(manifest->GetDatasetVersion());
  }
  versions |= actions::sort([](auto& v1, auto& v2) { return v1.version() < v2.version(); });
  return versions;
}

::arrow::Result<DatasetVersion> LanceDataset::latest_version() const {
  auto latest_version_path = GetManifestPath(impl_->base_uri);
  ARROW_ASSIGN_OR_RAISE(auto manifest, OpenManifest(impl_->fs, latest_version_path));
  return manifest->GetDatasetVersion();
}

::arrow::Result<std::shared_ptr<::arrow::dataset::Dataset>> LanceDataset::ReplaceSchema(
    [[maybe_unused]] std::shared_ptr<::arrow::Schema> schema) const {
  return std::make_shared<LanceDataset>(*this);
}

::arrow::Result<::arrow::dataset::FragmentIterator> LanceDataset::GetFragmentsImpl(
    [[maybe_unused]] ::arrow::compute::Expression predicate) {
  std::vector<std::shared_ptr<::arrow::dataset::Fragment>> fragments =
      impl_->manifest->fragments() | views::transform([this](auto& data_fragment) {
        return std::make_shared<LanceFragment>(
            impl_->fs, impl_->data_dir(), data_fragment, impl_->manifest->schema());
      }) |
      ranges::to<decltype(fragments)>;

  return ::arrow::MakeVectorIterator(fragments);
}

}  // namespace lance::arrow