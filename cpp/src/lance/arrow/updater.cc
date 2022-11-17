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
#include "lance/format/schema.h"
#include "lance/io/writer.h"

namespace fs = std::filesystem;

namespace lance::arrow {

class Updater::Impl {
 public:
  Impl(std::shared_ptr<LanceDataset> dataset,
       ::arrow::dataset::FragmentVector fragments,
       std::shared_ptr<lance::format::Schema> column_schema)
      : dataset_(std::move(dataset)),
        column_schema_(std::move(column_schema)),
        fragments_(std::move(fragments)),
        fragment_it_(fragments_.begin()) {}

  /// Copy constructor
  Impl(const Impl& other)
      : dataset_(other.dataset_),
        column_schema_(other.column_schema_),
        fragments_(other.fragments_.begin(), other.fragments_.end()),
        fragment_it_(fragments_.begin()) {}

  ::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> Next();

  ::arrow::Status Update(const std::shared_ptr<::arrow::Array>& arr);

 private:
  auto data_dir() const { return dataset_->impl_->data_dir(); }

  const auto& fs() const { return dataset_->impl_->fs; }

  /// Prepare to read the next fragment.
  ::arrow::Status NextFragment();

  std::shared_ptr<LanceDataset> dataset_;
  std::shared_ptr<lance::format::Schema> column_schema_;
  ::arrow::dataset::FragmentVector fragments_;

  // Used to store the updated fragments.
  ::arrow::dataset::FragmentVector updated_fragments_;

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
  ARROW_RETURN_NOT_OK(scan_builder->BatchSize(std::numeric_limits<int64_t>::max()));
  ARROW_ASSIGN_OR_RAISE(auto scanner, scan_builder->Finish());
  ARROW_ASSIGN_OR_RAISE(batch_generator_, (*fragment_it_)->ScanBatchesAsync(scanner->options()));

  return ::arrow::Status::OK();
}

::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> Updater::Impl::Next() {
  if (fragment_it_ == fragments_.end()) {
    return nullptr;
  }
  if (last_batch_) {
    return ::arrow::Status::IOError("Have not consumed/committed last batch");
  }
  if (!writer_) {
    ARROW_RETURN_NOT_OK(NextFragment());
  }

  auto fut = batch_generator_();
  ARROW_ASSIGN_OR_RAISE(auto batch, fut.result());
  if (!batch) {
    // EOL
    writer_.reset();
    batch_generator_ = nullptr;
  }
  return batch;
}

::arrow::Status Updater::Impl::Update(const std::shared_ptr<::arrow::Array>& arr) {
  // Sanity checks.
  if (arr->num_fields() != 1) {
    return ::arrow::Status::IOError(
        "Update only supports 1 column at a time, but got ", arr->num_fields(), " columns.");
  }
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

::arrow::Result<Updater> Updater::Make(std::shared_ptr<LanceDataset> dataset,
                                       const std::shared_ptr<::arrow::Field>& field) {
  auto arrow_schema = ::arrow::schema({field});
  ARROW_ASSIGN_OR_RAISE(auto full_schema, dataset->impl_->manifest->schema()->Merge(*arrow_schema));
  ARROW_ASSIGN_OR_RAISE(auto column_schema, full_schema->Project(*arrow_schema));
  ARROW_ASSIGN_OR_RAISE(auto fragment_iter, dataset->GetFragments());
  // Use vector to make implementation easier.
  // We can later to use FragmentIterator for datasets with a lot of Fragments.
  ARROW_ASSIGN_OR_RAISE(auto fragments, fragment_iter.ToVector());
  auto impl =
      std::make_unique<Impl>(std::move(dataset), std::move(fragments), std::move(column_schema));
  return Updater(std::move(impl));
}

::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> Updater::Next() { return impl_->Next(); }

::arrow::Status Updater::Update(const std::shared_ptr<::arrow::Array>& arr) {
  return impl_->Update(arr);
}

Updater::Updater(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

::arrow::Result<std::shared_ptr<LanceDataset>> Updater::Finish() {
  return ::arrow::Status::NotImplemented("not implemented");
}

}  // namespace lance::arrow