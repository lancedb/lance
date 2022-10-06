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

#pragma once

#include <arrow/array.h>
#include <arrow/result.h>
#include <arrow/table.h>
#include <arrow/type.h>

#include <memory>
#include <string>
#include <vector>

namespace lance::testing {

/// Build an Array from JSON payload.
::arrow::Result<std::shared_ptr<::arrow::Array>> ArrayFromJSON(
    const std::shared_ptr<::arrow::DataType>& type, const std::string& json);

/// Create an Arrow Table from batches (in JSON form).
///
/// \param schema Table schema
/// \param json an array of json payloads, each json payload represents one ::arrow::RecordBatch.
/// \return ::arrow::Table if success.
::arrow::Result<std::shared_ptr<::arrow::Table>> TableFromJSON(
    const std::shared_ptr<::arrow::Schema>& schema, const std::vector<std::string>& json);

::arrow::Result<std::shared_ptr<::arrow::Table>> TableFromJSON(
    const std::shared_ptr<::arrow::Schema>& schema, const std::string& json);

}  // namespace lance::testing
