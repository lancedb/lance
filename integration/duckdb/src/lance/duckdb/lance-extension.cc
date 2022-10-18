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

#include <duckdb.hpp>

#include "lance/duckdb/list_functions.h"
#include "lance/duckdb/ml/functions.h"
#include "lance/duckdb/vector_functions.h"

class LanceExtension : public ::duckdb::Extension {
 public:
  void Load(::duckdb::DuckDB &db) override {
    duckdb::Connection con(db);
    con.BeginTransaction();
    auto &context = *con.context;
    auto &catalog = ::duckdb::Catalog::GetCatalog(context);

    for (auto &func : lance::duckdb::GetListFunctions()) {
      catalog.CreateFunction(context, func.get());
    }

    for (auto &func : lance::duckdb::GetVectorFunctions()) {
      catalog.CreateFunction(context, func.get());
    }

    for (auto &func : lance::duckdb::ml::GetMLFunctions()) {
      catalog.CreateFunction(context, func.get());
    }

    for (auto &func : lance::duckdb::ml::GetMLTableFunctions()) {
      catalog.CreateTableFunction(context, func.get());
    }

    con.Commit();
  }

  std::string Name() override { return std::string("lance"); }
};

extern "C" {

DUCKDB_EXTENSION_API void lance_init(duckdb::DatabaseInstance &db) {
  duckdb::DuckDB db_wrapper(db);
  db_wrapper.LoadExtension<LanceExtension>();
}

DUCKDB_EXTENSION_API const char *lance_version() { return duckdb::DuckDB::LibraryVersion(); }
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
