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

#include <arrow/array.h>
#include <arrow/array/concatenate.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/type_traits.h>
#include <fmt/format.h>

#include <algorithm>
#include <filesystem>
#include <mutex>
#include <range/v3/all.hpp>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "lance/arrow/dataset_ext.h"
#include "lance/arrow/file_lance.h"
#include "lance/arrow/fragment.h"
#include "lance/arrow/hash_merger.h"
#include "lance/arrow/updater.h"
#include "lance/arrow/utils.h"
#include "lance/format/manifest.h"
#include "lance/format/schema.h"
#include "lance/io/reader.h"
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

std::string GetBasenameTemplate() { return GetUUIDString() + "_{i}.lance"; }

::arrow::Result<std::shared_ptr<lance::format::Manifest>> OpenManifest(
    const std::shared_ptr<::arrow::fs::FileSystem>& fs, const std::string& path) {
  ARROW_ASSIGN_OR_RAISE(auto in, fs->OpenInputFile(path));
  return lance::io::FileReader::OpenManifest(in);
}

::arrow::Status CollectDictionary(const std::shared_ptr<lance::format::Field>& field,
                                  const std::shared_ptr<::arrow::Array>& arr) {
  assert(field && arr);
  assert(field->type()->Equals(arr->type()));
  auto data_type = field->type();
  if (::arrow::is_dictionary(data_type->id())) {
    auto dict_arr = std::dynamic_pointer_cast<::arrow::DictionaryArray>(arr);
    return field->set_dictionary(dict_arr->dictionary());
  }

  if (is_list(data_type)) {
    auto list_arr = std::dynamic_pointer_cast<::arrow::ListArray>(arr);
    ARROW_RETURN_NOT_OK(CollectDictionary(field->field(0), list_arr->values()));
  } else if (is_struct(data_type)) {
    auto struct_arr = std::dynamic_pointer_cast<::arrow::StructArray>(arr);
    for (auto& child : field->fields()) {
      auto child_arr = struct_arr->GetFieldByName(child->name());
      if (child_arr == nullptr) {
        return ::arrow::Status::Invalid("CollectDictionary: schema mismatch: field ",
                                        child->name(),
                                        "does not exist in the table: ",
                                        struct_arr->type());
      }
      ARROW_RETURN_NOT_OK(CollectDictionary(child, child_arr));
    }
  }
  return ::arrow::Status::OK();
}

::arrow::Status CollectDictionary(const std::shared_ptr<lance::format::Schema>& schema,
                                  const std::shared_ptr<::arrow::dataset::Scanner>& scanner) {
  ARROW_ASSIGN_OR_RAISE(auto example, scanner->Head(1));
  if (example->num_rows() == 0) {
    return ::arrow::Status::Invalid("CollectDictionary: empty dataset");
  }
  for (auto& field : schema->fields()) {
    auto chunked_arr = example->GetColumnByName(field->name());
    if (chunked_arr == nullptr) {
      return ::arrow::Status::Invalid("CollectDictionary: schema mismatch: field ",
                                      field->name(),
                                      "does not exist in the table: ",
                                      example->schema());
    }
    assert(chunked_arr->num_chunks() > 0);
    ARROW_RETURN_NOT_OK(CollectDictionary(field, chunked_arr->chunk(0)));
  }
  return ::arrow::Status::OK();
}

}  // namespace

DatasetVersion::DatasetVersion(uint64_t version,
                               std::chrono::time_point<std::chrono::system_clock> created)
    : version_(version), timestamp_(created) {}

uint64_t DatasetVersion::version() const { return version_; }

const std::chrono::time_point<std::chrono::system_clock>& DatasetVersion::timestamp() const {
  return timestamp_;
}

std::time_t DatasetVersion::timet_timestamp() const {
  return std::chrono::system_clock::to_time_t(timestamp_);
}

DatasetVersion& DatasetVersion::operator++() {
  version_++;
  Touch();
  return *this;
}

DatasetVersion DatasetVersion::operator++(int) {
  version_++;
  Touch();
  return *this;
}

void DatasetVersion::Touch() { timestamp_ = std::chrono::system_clock::now(); }

//-------------------------
// LanceDataset::Impl
//-------------------------

std::string LanceDataset::Impl::data_dir() const { return fs::path(base_uri) / kDataDir; }

std::string LanceDataset::Impl::versions_dir() const { return fs::path(base_uri) / kVersionsDir; }

namespace {

::arrow::Status WriteManifest(const std::shared_ptr<::arrow::fs::FileSystem>& fs,
                              const std::string& base_uri,
                              const std::shared_ptr<format::Manifest>& manifest) {
  // Write the manifest version file.
  // It only supports single writer at the moment.
  auto version_dir = (fs::path(base_uri) / kVersionsDir).string();
  ARROW_RETURN_NOT_OK(fs->CreateDir(version_dir));
  auto manifest_path = GetManifestPath(base_uri, manifest->version());
  {
    // Keep the serialized time, which is when this particular version of dataset is visible.
    manifest->Touch();
    ARROW_ASSIGN_OR_RAISE(auto out, fs->OpenOutputStream(manifest_path));
    ARROW_RETURN_NOT_OK(lance::io::FileWriter::WriteManifest(out, *manifest));
  }
  auto latest_manifest_path = GetManifestPath(base_uri, std::nullopt);
  return fs->CopyFile(manifest_path, latest_manifest_path);
}

}  // namespace

::arrow::Result<std::unique_ptr<LanceDataset::Impl>> LanceDataset::Impl::WriteNewVersion(
    std::shared_ptr<lance::format::Manifest> new_manifest) const {
  ARROW_RETURN_NOT_OK(WriteManifest(fs, base_uri, new_manifest));
  return std::make_unique<Impl>(fs, base_uri, std::move(new_manifest));
}

//---------------------------
LanceDataset::LanceDataset(std::unique_ptr<LanceDataset::Impl> impl)
    : ::arrow::dataset::Dataset(impl->manifest->schema()->ToArrow()), impl_(std::move(impl)) {}

LanceDataset::LanceDataset(const LanceDataset& other)
    : LanceDataset(std::make_unique<Impl>(*other.impl_)) {}

LanceDataset::~LanceDataset() {}

::arrow::Status LanceDataset::Write(const ::arrow::dataset::FileSystemDatasetWriteOptions& options,
                                    const std::shared_ptr<::arrow::dataset::Dataset>& dataset,
                                    WriteMode mode) {
  ARROW_ASSIGN_OR_RAISE(auto scan_builder, dataset->NewScan());
  ARROW_ASSIGN_OR_RAISE(auto scanner, scan_builder->Finish());
  return Write(options, std::move(scanner), mode);
}

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
      auto existing_arrow_schema = existing_manifest->schema()->ToArrow();

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
  ARROW_RETURN_NOT_OK(CollectDictionary(manifest->schema(), scanner));

  // Write manifest file
  auto lance_option = options;
  lance_option.base_dir = data_dir;
  lance_option.existing_data_behavior = ::arrow::dataset::ExistingDataBehavior::kOverwriteOrIgnore;
  lance_option.create_dir = true;
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

  manifest->AppendFragments(CreateFragments(paths, *manifest->schema()));

  return WriteManifest(fs, base_dir, manifest);
}

::arrow::Result<std::shared_ptr<LanceDataset>> LanceDataset::Make(const std::string& uri,
                                                                  std::optional<uint64_t> version) {
  std::string path;
  ARROW_ASSIGN_OR_RAISE(auto fs, ::arrow::fs::FileSystemFromUriOrPath(uri, &path));
  return Make(fs, path, version);
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
  auto result = OpenManifest(fs, manifest_path);

  if (result.status().IsIOError() && result.status().message().starts_with("Path does not exist")) {
    if (!version) {
      return ::arrow::Status::IOError("Can not find the latest version of the dataset");
    }
    return ::arrow::Status::IOError("Version ", version.value(), " does not exist");
  }

  ARROW_ASSIGN_OR_RAISE(auto manifest, result);
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

DatasetVersion LanceDataset::version() const { return impl_->manifest->GetDatasetVersion(); }

::arrow::Result<std::shared_ptr<UpdaterBuilder>> LanceDataset::NewUpdate(
    const std::shared_ptr<::arrow::Field>& new_field) const {
  return NewUpdate(::arrow::schema({new_field}));
}

::arrow::Result<std::shared_ptr<UpdaterBuilder>> LanceDataset::NewUpdate(
    const std::shared_ptr<::arrow::Schema>& new_columns) const {
  return std::make_shared<UpdaterBuilder>(std::make_shared<LanceDataset>(*this), new_columns);
}

::arrow::Result<std::shared_ptr<LanceDataset>> LanceDataset::AddColumn(
    const std::shared_ptr<::arrow::Field>& field, ::arrow::compute::Expression expression) {
  if (!expression.IsScalarExpression()) {
    return ::arrow::Status::Invalid(
        "LanceDataset::AddColumn: expression is not a scalar expression.");
  }
  ARROW_ASSIGN_OR_RAISE(expression, expression.Bind(*schema()));
  ARROW_ASSIGN_OR_RAISE(auto builder, NewUpdate(field));
  if (::arrow::compute::ExpressionHasFieldRefs(expression)) {
    std::vector<std::string> columns;
    for (const auto& ref : ::arrow::compute::FieldsInExpression(expression)) {
      columns.emplace_back(arrow::ToColumnName(ref));
    }
    builder->Project(columns);
  }
  ARROW_ASSIGN_OR_RAISE(auto updater, builder->Finish());

  while (true) {
    ARROW_ASSIGN_OR_RAISE(auto batch, updater->Next());
    if (!batch) {
      break;
    }

#ifndef NDEBUG
    // Due to lack of injection point to test schema in the unit tests, let's do assert here.
    // Assert will be disabled in the release build.
    if (::arrow::compute::ExpressionHasFieldRefs(expression)) {
      assert(batch->schema()->Equals(
          impl_->manifest->schema()->Project(expression).ValueOrDie()->ToArrow()));
    }
#endif

    ARROW_ASSIGN_OR_RAISE(auto datum,
                          ::arrow::compute::ExecuteScalarExpression(expression, *schema(), batch));
    std::shared_ptr<::arrow::Array> arr;
    if (datum.is_scalar()) {
      ARROW_ASSIGN_OR_RAISE(arr, CreateArray(datum.scalar(), batch->num_rows()));
    } else if (datum.is_chunked_array()) {
      auto chunked_arr = datum.chunked_array();
      ARROW_ASSIGN_OR_RAISE(arr, ::arrow::Concatenate(chunked_arr->chunks()));
    } else {
      arr = datum.make_array();
    }
    ARROW_RETURN_NOT_OK(updater->UpdateBatch(arr));
  }
  return updater->Finish();
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
            impl_->fs, impl_->data_dir(), data_fragment, impl_->manifest);
      }) |
      ranges::to<decltype(fragments)>;

  return ::arrow::MakeVectorIterator(fragments);
}

::arrow::Result<std::shared_ptr<LanceDataset>> LanceDataset::Merge(
    const std::shared_ptr<::arrow::Table>& other,
    const std::string& on,
    ::arrow::MemoryPool* pool) {
  return Merge(other, on, on, pool);
}

::arrow::Result<std::shared_ptr<LanceDataset>> LanceDataset::Merge(
    const std::shared_ptr<::arrow::Table>& right,
    const std::string& left_on,
    const std::string& right_on,
    ::arrow::MemoryPool* pool) {
  /// Sanity checks
  auto left_column = schema_->GetFieldByName(left_on);
  if (left_column == nullptr) {
    return ::arrow::Status::Invalid(
        fmt::format("Column {} does not exist in the dataset.", left_on));
  }
  auto right_column = right->GetColumnByName(right_on);
  if (right_column == nullptr) {
    return ::arrow::Status::Invalid(
        fmt::format("Column {} does not exist in the table.", right_on));
  }

  auto& left_type = left_column->type();
  auto& right_type = right_column->type();
  if (!left_type->Equals(right_type)) {
    return ::arrow::Status::Invalid("LanceDataset::AddColumns: types are not equal: ",
                                    left_type->ToString(),
                                    " != ",
                                    right_type->ToString());
  }

  // First phase, build hash table (in memory for simplicity)
  auto merger = HashMerger(right, right_on, pool);
  ARROW_RETURN_NOT_OK(merger.Init());

  // Second phase
  auto table_schema = right->schema();
  ARROW_ASSIGN_OR_RAISE(auto incoming_schema,
                        table_schema->RemoveField(table_schema->GetFieldIndex(right_on)));
  ARROW_ASSIGN_OR_RAISE(auto update_builder, NewUpdate(std::move(incoming_schema)));
  update_builder->Project({left_on});
  ARROW_ASSIGN_OR_RAISE(auto updater, update_builder->Finish());

  while (true) {
    ARROW_ASSIGN_OR_RAISE(auto batch, updater->Next());
    if (!batch) {
      break;
    }
    assert(batch->schema()->Equals(::arrow::schema({left_column})));
    auto index_arr = batch->GetColumnByName(left_on);
    ARROW_ASSIGN_OR_RAISE(auto right_batch, merger.Collect(index_arr));
    ARROW_RETURN_NOT_OK(updater->UpdateBatch(right_batch));
  }
  return updater->Finish();
}

}  // namespace lance::arrow