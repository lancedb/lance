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

#include "lance/duckdb/lance_reader.h"

#include <memory>

namespace lance::duckdb {

struct GlobalLanceReaderFunctionState : public ::duckdb::GlobalTableFunctionState {};

struct LocalLanceReaderFunctionState : public ::duckdb::LocalTableFunctionState {

};

void LanceScan(::duckdb::ClientContext &context,
               ::duckdb::TableFunctionInput &input,
               ::duckdb::DataChunk &output) {}

::duckdb::TableFunctionSet GetLanceReaderFunction() {
  ::duckdb::TableFunctionSet func_set("lance_scan");

  ::duckdb::TableFunction table_function({::duckdb::LogicalType::VARCHAR}, LanceScan);
  table_function.projection_pushdown = true;
  table_function.filter_pushdown = true;
  table_function.filter_prune = true;

  func_set.AddFunction(table_function);
  return func_set;
}

}  // namespace lance::duckdb