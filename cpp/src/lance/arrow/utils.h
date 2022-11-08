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
#include <arrow/dataset/dataset.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>

#include <memory>
#include <vector>

#include "lance/format/schema.h"

namespace lance::arrow {

/// Merge two same-length record batches into one.
::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> MergeRecordBatches(
    const std::shared_ptr<::arrow::RecordBatch>& lhs,
    const std::shared_ptr<::arrow::RecordBatch>& rhs,
    ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

/// Merge a list of record batches that represent the different columns of the same rows,
/// into a single record batch.
///
/// \param batches A list of record batches. Must have the same length.
/// \param pool memory pool.
/// \return the merged record batch. Or nullptr if batches is empty.
::arrow::Result<std::shared_ptr<::arrow::RecordBatch>> MergeRecordBatches(
    const std::vector<std::shared_ptr<::arrow::RecordBatch>>& batches,
    ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

::arrow::Result<std::shared_ptr<::arrow::StructArray>> MergeStructArrays(
    const std::shared_ptr<::arrow::StructArray>& lhs,
    const std::shared_ptr<::arrow::StructArray>& rhs,
    ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

/// Merge two Arrow Schemas if they are compatible.
::arrow::Result<std::shared_ptr<::arrow::Schema>> MergeSchema(const ::arrow::Schema& lhs,
                                                              const ::arrow::Schema& rhs);

/// Open Lance dataset from URI.
::arrow::Result<std::shared_ptr<::arrow::dataset::FileSystemDataset>> OpenDataset(
    const std::string& uri, std::shared_ptr<::arrow::dataset::Partitioning> partitioning = nullptr);

}  // namespace lance::arrow
