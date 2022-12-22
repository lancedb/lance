// Copyright 2022 Lance Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma once

/// \brief Lance Core Adaptors and utilities

#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/type_fwd.h>

#include <duckdb/common/exception.hpp>
#include <duckdb/common/types.hpp>

namespace lance::duckdb {

template <typename T, typename E = ::duckdb::IOException>
T GetResult(::arrow::Result<T>&& result) {
  if (result.ok()) {
    return std::move(result.ValueOrDie());
  }
  throw E(result.status().message());
}

template <typename E = ::duckdb::IOException>
void CheckStatus(const ::arrow::Status& status) {
  if (!status.ok()) {
    throw E(status.message());
  }
}

/// Convert Arrow and Lance types into DuckDB logical type
::duckdb::LogicalType ToLogicalType(const ::arrow::DataType& arrow_type);

}  // namespace lance::duckdb
