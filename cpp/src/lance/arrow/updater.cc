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

#include "lance/arrow/updater.h"

#include <arrow/dataset/type_fwd.h>
#include <arrow/result.h>
#include <fmt/format.h>

#include <filesystem>
#include <limits>
#include <memory>
#include <vector>

#include "lance/arrow/dataset.h"
#include "lance/arrow/dataset_ext.h"
#include "lance/arrow/file_lance.h"
#include "lance/arrow/fragment.h"
#include "lance/arrow/utils.h"
#include "lance/format/data_fragment.h"
#include "lance/format/schema.h"
#include "lance/io/writer.h"

namespace fs = std::filesystem;

namespace lance::arrow {

class Updater::Impl {
 public:
  Impl(std::shared_ptr<LanceDataset> dataset,
       ::arrow::dataset::FragmentVector fragments,
       std::shared_ptr<lance::format::Schema> full_schema,
       std::shared_ptr<lance::format::Schema> column_schema,
       std::vector<std::string> projection_columns)
      : dataset_(std::move(dataset)),
        full_schema_(std::move(full_schema)),
        column_schema_(std::move(column_schema)),
        fragments_(std::move(fragments)),
        projected_columns_(std::move(projection_columns)),
        fragment_it_(fragments_.begin()) {}

  /// Copy constructor
  Impl(const Impl& other)
      : dataset_(other.dataset_),
        full_schema_(other.full_schema_),
        column_schema_(other.column_schema_),
        fragments_(other.fragments_.begin(), other.fragments_.end()),
        projected_columns_(other.projected_columns_.begin(), other.projected_columns_.end()),
        fragment_it_(fragments_.begin()) {}

  ::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> Next();

  ::arrow::Status UpdateBatch(const std::shared_ptr<::arrow::Array>& arr);

  ::arrow::Result<std::shared_ptr<LanceDataset>> Finish();

 private:
  auto data_dir() const { return dataset_->impl_->data_dir(); }

  const auto& fs() const { return dataset_->impl_->fs; }

  /// Prepare to read the next fragment.
  ::arrow::Status NextFragment();

  std::shared_ptr<LanceDataset> dataset_;
  /// The schema of the newly generated dataset.
  std::shared_ptr<lance::format::Schema> full_schema_;
  /// The schema of the column to updated/added
  std::shared_ptr<lance::format::Schema> column_schema_;
  /// A copy of fragments.
  ::arrow::dataset::FragmentVector fragments_;

  std::vector<std::string> projected_columns_;

  // Used to store the updated fragments.
  std::vector<std::shared_ptr<format::DataFragment>> data_fragments_;

  // Track runtime information.

  ::arrow::dataset::FragmentVector::iterator fragment_it_;
  std::shared_ptr<::arrow::RecordBatch> last_batch_;
  std::unique_ptr<lance::io::FileWriter> writer_;
  ::arrow::RecordBatchGenerator batch_generator_;
};

::arrow::Status Updater::Impl::NextFragment() {
  assert(!writer_);
  auto& fragment = *fragment_it_;
  assert(fragment->type_name() == "lance");
  auto lance_fragment = std::dynamic_pointer_cast<LanceFragment>(fragment);
  assert(lance_fragment);

  std::string file_path = fs::path(data_dir()) / fmt::format("{}.lance", GetUUIDString());
  ARROW_ASSIGN_OR_RAISE(auto output, fs()->OpenOutputStream(file_path));
  auto write_options = LanceFileFormat().DefaultWriteOptions();
  writer_ =
      std::make_unique<io::FileWriter>(column_schema_, std::move(write_options), std::move(output));

  ARROW_ASSIGN_OR_RAISE(auto scan_builder, dataset_->NewScan());
  if (!projected_columns_.empty()) {
    ARROW_RETURN_NOT_OK(scan_builder->Project(projected_columns_));
  }
  //  ARROW_RETURN_NOT_OK(scan_builder->BatchSize(std::numeric_limits<int64_t>::max()));
  ARROW_ASSIGN_OR_RAISE(auto scanner, scan_builder->Finish());
  ARROW_ASSIGN_OR_RAISE(batch_generator_, (*fragment_it_)->ScanBatchesAsync(scanner->options()));

  // Track the new data files
  std::vector<format::DataFile> data_files(lance_fragment->data_fragment()->data_files());
  data_files.emplace_back(::lance::format::DataFile({file_path, column_schema_->GetFieldIds()}));
  auto new_fragment = std::make_shared<format::DataFragment>(data_files);
  data_fragments_.emplace_back(std::move(new_fragment));

  return ::arrow::Status::OK();
}

::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> Updater::Impl::Next() {
  if (fragment_it_ == fragments_.end()) {
    return nullptr;
  }
  if (last_batch_) {
    return ::arrow::Status::IOError("Have not consumed/committed last batch");
  }
  while (!last_batch_ && fragment_it_ != fragments_.end()) {
    if (!writer_) {
      ARROW_RETURN_NOT_OK(NextFragment());
    }

    auto fut = batch_generator_();
    ARROW_ASSIGN_OR_RAISE(last_batch_, fut.result());
    if (!last_batch_) {
      ARROW_RETURN_NOT_OK(writer_->Finish().result());
      // EOF
      writer_.reset();
      ++fragment_it_;
    }
  }

  return last_batch_;
}

::arrow::Status Updater::Impl::UpdateBatch(const std::shared_ptr<::arrow::Array>& arr) {
  // Sanity checks.
  if (!last_batch_) {
    return ::arrow::Status::IOError(
        "Did not read batch before update, did you call Updater::Next() before?");
  }
  if (last_batch_->num_rows() != arr->length()) {
    return ::arrow::Status::IOError(
        fmt::format("Updater::Update: input size({}) != output size({})",
                    last_batch_->num_rows(),
                    arr->length()));
  }

  assert(writer_);
  last_batch_.reset();
  auto batch = ::arrow::RecordBatch::Make(column_schema_->ToArrow(), arr->length(), {arr});
  return writer_->Write(batch);
}

::arrow::Result<std::shared_ptr<LanceDataset>> Updater::Impl::Finish() {
  if (fragment_it_ != fragments_.end()) {
    return ::arrow::Status::Invalid("Updater::Finish: there are remaining data to consume.");
  }
  ARROW_ASSIGN_OR_RAISE(auto latest_version, dataset_->latest_version());
  ++latest_version;
  auto new_manifest =
      std::make_shared<lance::format::Manifest>(full_schema_, latest_version, data_fragments_);
  ARROW_ASSIGN_OR_RAISE(auto dataset_impl, dataset_->impl_->WriteNewVersion(new_manifest));
  return std::shared_ptr<LanceDataset>(new LanceDataset(std::move(dataset_impl)));
}

Updater::~Updater() {}

::arrow::Result<std::shared_ptr<Updater>> Updater::Make(
    std::shared_ptr<LanceDataset> dataset,
    const std::shared_ptr<::arrow::Field>& field,
    const std::vector<std::string>& projection_columns) {
  auto arrow_schema = ::arrow::schema({field});
  ARROW_ASSIGN_OR_RAISE(auto full_schema, dataset->impl_->manifest->schema()->Merge(*arrow_schema));
  ARROW_ASSIGN_OR_RAISE(auto column_schema, full_schema->Project(*arrow_schema));
  ARROW_ASSIGN_OR_RAISE(auto fragment_iter, dataset->GetFragments());
  // Use vector to make implementation easier.
  // We can later to use FragmentIterator for datasets with a lot of Fragments.
  ARROW_ASSIGN_OR_RAISE(auto fragments, fragment_iter.ToVector());
  auto impl = std::make_unique<Impl>(std::move(dataset),
                                     std::move(fragments),
                                     std::move(full_schema),
                                     std::move(column_schema),
                                     projection_columns);
  return std::shared_ptr<Updater>(new Updater(std::move(impl)));
}

::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> Updater::Next() { return impl_->Next(); }

::arrow::Status Updater::UpdateBatch(const std::shared_ptr<::arrow::Array>& arr) {
  return impl_->UpdateBatch(arr);
}

Updater::Updater(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

::arrow::Result<std::shared_ptr<LanceDataset>> Updater::Finish() { return impl_->Finish(); }

UpdaterBuilder::UpdaterBuilder(std::shared_ptr<LanceDataset> source,
                               std::shared_ptr<::arrow::Field> field)
    : dataset_(std::move(source)), field_(std::move(field)) {}

void UpdaterBuilder::Project(std::vector<std::string> columns) {
  projection_columns_ = std::move(columns);
}

::arrow::Result<std::shared_ptr<Updater>> UpdaterBuilder::Finish() {
  return Updater::Make(dataset_, field_, projection_columns_);
}

}  // namespace lance::arrow