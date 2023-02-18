//  Copyright 2023 Lance Authors
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

#include "duckdb_ext.h"

#include <string>

#include "duckdb.hpp"

namespace {

auto build_child_list(idx_t n_pairs, const char *const *names, duckdb_logical_type const *types) {
  duckdb::child_list_t<duckdb::LogicalType> members;
  for (idx_t i = 0; i < n_pairs; i++) {
    members.emplace_back(std::string(names[i]), *(duckdb::LogicalType *)types[i]);
  }
  return members;
}

}  // namespace

extern "C" {

duckdb_logical_type duckdb_create_struct_type(idx_t n_pairs,
                                              const char **names,
                                              const duckdb_logical_type *types) {
  auto *stype = new duckdb::LogicalType;
  *stype = duckdb::LogicalType::STRUCT(build_child_list(n_pairs, names, types));
  return reinterpret_cast<duckdb_logical_type>(stype);
}

}