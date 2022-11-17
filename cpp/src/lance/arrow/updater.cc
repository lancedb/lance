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
#include <vector>

#include "lance/arrow/dataset.h"
#include "lance/arrow/dataset_ext.h"
#include "lance/arrow/fragment.h"
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
        fragment_it_(fragments.begin()) {}

  /// Copy constructor
  explicit Impl(const Impl& other)
      : dataset_(other.dataset_),
        column_schema_(other.column_schema_),
        fragments_(other.fragments_.begin(), other.fragments_.end()),
        fragment_it_(fragments_.begin()) {}

  ::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> Next();

  ::arrow::Status Update(const std::shared_ptr<::arrow::Array>& arr);

 private:
  std::shared_ptr<LanceDataset> dataset_;
  std::shared_ptr<lance::format::Schema> column_schema_;
  ::arrow::dataset::FragmentVector fragments_;
  ::arrow::dataset::FragmentVector::iterator fragment_it_;
  // Used to store the updated fragments.
  ::arrow::dataset::FragmentVector updated_fragments_;
  std::shared_ptr<::arrow::RecordBatch> last_batch_;
  std::unique_ptr<lance::io::FileWriter> writer_;
};

::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> Updater::Impl::Next() {
  if (fragment_it_ == fragments_.end()) {
    return nullptr;
  }
  if (last_batch_) {
    return ::arrow::Status::IOError("Have not consumed/committed last batch");
  }
  if (!writer_) {
    auto& fragment = *fragment_it_;
    assert(fragment->type_name() == "lance");
    auto lance_fragment = std::dynamic_pointer_cast<LanceFragment>(fragment);
    assert(lance_fragment);
    // TODO: open new writer for each fragment
  }

  // TODO: read the next batch
  // TODO: if reaches to the end of a fragment, close the writer, and reopen another writer.
  // TODO: return RecordBatch

  return ::arrow::Result<std::shared_ptr<::arrow::RecordBatch>>();
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

  last_batch_.reset();
  return ::arrow::Status::NotImplemented("not implemented");
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