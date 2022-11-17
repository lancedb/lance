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

#pragma once

#include <arrow/array.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>
#include <arrow/status.h>

#include <memory>

#include "lance/arrow/dataset.h"

namespace lance::arrow {

/// \brief Streaming Updater
///
/// Experimental API
class Updater {
 public:
  static ::arrow::Result<Updater> Make(std::shared_ptr<LanceDataset> dataset);

  /// Returns the next batch as updater inputs. Return `nullptr` for the end of dataset.
  ::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> Next();

  /// Update a new column
  ::arrow::Status Update(const std::shared_ptr<::arrow::Array>& arr);

  /// Finish the update and returns a new version of dataset.
  ::arrow::Result<std::shared_ptr<LanceDataset>> Finish();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;

  Updater(std::unique_ptr<Impl> impl);
};

}  // namespace lance::arrow