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

#include <vector>

#include "lance/arrow/dataset.h"
#include "lance/arrow/fragment.h"

namespace lance::arrow {

class Updater::Impl {
 public:
  Impl(std::shared_ptr<LanceDataset> dataset, ::arrow::dataset::FragmentIterator iter)
      : dataset_(std::move(dataset)), fragment_iter_(std::move(iter)) {}

  ::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> Next();

  ::arrow::Status Update(const std::shared_ptr<::arrow::Array>& arr);

 private:
  std::shared_ptr<LanceDataset> dataset_;
  ::arrow::dataset::FragmentIterator fragment_iter_;
  std::shared_ptr<::arrow::RecordBatch> last_batch_;
};

::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> Updater::Impl::Next() {
  return ::arrow::Result<std::shared_ptr<::arrow::RecordBatch>>();
}

::arrow::Status Updater::Impl::Update(const std::shared_ptr<::arrow::Array>& arr) {
  if (arr->num_fields() != 1) {
    return ::arrow::Status::IOError("Update only supports 1 column at a time, but got ",
                                    arr->num_fields());
  }
  return ::arrow::Status::NotImplemented("not implemented");
}

::arrow::Result<Updater> Updater::Make(std::shared_ptr<LanceDataset> dataset) {
  ARROW_ASSIGN_OR_RAISE(auto fragment_it, dataset->GetFragments());
  auto impl = std::make_unique<Impl>(std::move(dataset), std::move(fragment_it));
  return Updater(std::move(impl));
}

::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> Updater::Next() { return impl_->Next(); }

::arrow::Status Updater::Update(const std::shared_ptr<::arrow::Array>& arr) {
  return impl_->Update(arr);
}

Updater::Updater(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

}  // namespace lance::arrow