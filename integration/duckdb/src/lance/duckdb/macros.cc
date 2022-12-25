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

#include "lance/duckdb/macros.h"

#include <duckdb/function/scalar_macro_function.hpp>
#include <duckdb/parser/parsed_data/create_macro_info.hpp>
#include <duckdb/parser/parser.hpp>
#include <string>
#include <unordered_map>

namespace lance::duckdb {

namespace {

std::unique_ptr<::duckdb::CreateMacroInfo> CreateMacro(
    const std::string& name,
    const std::vector<std::string>& arguments,
    const std::string& macro,
    const std::unordered_map<std::string, std::string>& default_arguments = {}) {
  auto expression = ::duckdb::Parser::ParseExpressionList(macro);
  assert(expression.size() == 1);
  auto info = std::make_unique<::duckdb::CreateMacroInfo>();

  auto macro_func = std::make_unique<::duckdb::ScalarMacroFunction>(std::move(expression[0]));
  for (auto& argument : arguments) {
    macro_func->parameters.emplace_back(std::make_unique<::duckdb::ColumnRefExpression>(argument));
  }

  for (auto& [name, expr] : default_arguments) {
    auto value_expr = ::duckdb::Parser::ParseExpressionList(expr);
    assert(value_expr.size() == 1);
    macro_func->default_parameters.emplace(name, std::move(value_expr[0]));
  }

  auto macro_info = std::make_unique<::duckdb::CreateMacroInfo>();
  macro_info->schema = DEFAULT_SCHEMA;
  macro_info->name = name;
  macro_info->temporary = true;
  macro_info->internal = true;
  macro_info->type = ::duckdb::CatalogType::MACRO_ENTRY;
  macro_info->function = std::move(macro_func);

  return macro_info;
}

}  // namespace

std::vector<std::unique_ptr<::duckdb::CreateFunctionInfo>> GetMacroFunctions() {
  decltype(GetMacroFunctions()) functions;

  functions.emplace_back(
      CreateMacro("dydx",
                  {"y", "x"},
                  "y - lag(y, 1) OVER (ORDER BY x) / (x - lag(x, 1, 0) OVER (ORDER BY x))"));
  functions.emplace_back(
      CreateMacro("window_all",
                  {"expr", "before", "after"},
                  "bool_and(expr) OVER (ROWS BETWEEN before PRECEDING AND after FOLLOWING)"));
  functions.emplace_back(
      CreateMacro("window_any",
                  {"expr", "before", "after"},
                  "bool_or(expr) OVER (ROWS BETWEEN before PRECEDING AND after FOLLOWING)"));

  return functions;
}

}  // namespace lance::duckdb