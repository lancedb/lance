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

#define DUCKDB_EXTENSION_MAIN

#include "lance-extension.h"

#include <duckdb.hpp>
#include <duckdb/catalog/default/default_functions.hpp>
#include <duckdb/parser/parsed_data/create_table_function_info.hpp>

#include "lance/duckdb/lance_reader.h"
#include "lance/duckdb/list_functions.h"
#if defined(WITH_PYTORCH)
#include "lance/duckdb/ml/functions.h"
#endif
#include "lance/duckdb/vector_functions.h"

namespace duckdb {

static DefaultMacro macros[] = {{DEFAULT_SCHEMA,
                                 "dydx",
                                 {"y", "x", nullptr},
                                 "y - lag(y, 1) OVER (ORDER BY x) / (x - lag(x, 1, 0) OVER (ORDER BY x))"},
                                {nullptr, nullptr, {nullptr}, nullptr}};

void LanceExtension::Load(::duckdb::DuckDB &db) {
  duckdb::Connection con(db);
  con.BeginTransaction();
  auto &context = *con.context;
  auto &catalog = ::duckdb::Catalog::GetCatalog(context);
  auto &config = DBConfig::GetConfig(*db.instance);

  for (auto &func : lance::duckdb::GetListFunctions()) {
    catalog.CreateFunction(context, func.get());
  }

  for (auto &func : lance::duckdb::GetVectorFunctions()) {
    catalog.CreateFunction(context, func.get());
  }

  for (idx_t index = 0; macros[index].name != nullptr; index++) {
    auto info = DefaultFunctionGenerator::CreateInternalMacroInfo(macros[index]);
    catalog.CreateFunction(*con.context, info.get());
  }

#if defined(WITH_PYTORCH)
  for (auto &func : lance::duckdb::ml::GetMLFunctions()) {
    catalog.CreateFunction(context, func.get());
  }

  for (auto &func : lance::duckdb::ml::GetMLTableFunctions()) {
    catalog.CreateTableFunction(context, func.get());
  }
#endif

  auto scan_func = lance::duckdb::GetLanceReaderFunction();
  ::duckdb::CreateTableFunctionInfo scan(scan_func);
  catalog.CreateTableFunction(context, &scan);

  config.replacement_scans.emplace_back(lance::duckdb::LanceScanReplacement);

  con.Commit();
}

std::string LanceExtension::Name() { return {"lance"}; }
};  // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void lance_init(duckdb::DatabaseInstance &db) {
  duckdb::DuckDB db_wrapper(db);
  db_wrapper.LoadExtension<duckdb::LanceExtension>();
}

DUCKDB_EXTENSION_API const char *lance_version() { return duckdb::DuckDB::LibraryVersion(); }
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
