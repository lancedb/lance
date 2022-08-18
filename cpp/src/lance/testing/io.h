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

#include <arrow/result.h>
#include <arrow/table.h>

#include <memory>

#include "lance/io/reader.h"

namespace lance::testing {

/// Make lance::io::FileReader from an Arrow Table.
::arrow::Result<std::shared_ptr<lance::io::FileReader>> MakeReader(
    const std::shared_ptr<::arrow::Table>& table);

}  // namespace lance::testing
