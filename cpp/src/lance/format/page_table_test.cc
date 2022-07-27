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

#include "lance/format/page_table.h"

#include <arrow/io/api.h>

#include <catch2/catch_test_macros.hpp>

using lance::format::PageTable;

TEST_CASE("Serialize page length") {
  lance::format::PageTable lt;

  int num_columns = 3;
  int num_batches = 5;
  for (int col = 0; col < num_columns; col++) {
    for (int batch = 0; batch < num_batches; batch++) {
      lt.SetPageInfo(col, batch, col * 10 + batch, col * 10 + batch);
    }
  }

  auto out_buf = arrow::io::BufferOutputStream::Create().ValueOrDie();
  lt.Write(out_buf).ValueOrDie();

  auto in_buf = std::make_shared<arrow::io::BufferReader>(out_buf->Finish().ValueOrDie());
  auto actual = PageTable::Make(in_buf, 0, num_columns, num_batches).ValueOrDie();

  for (int col = 0; col < num_columns; col++) {
    for (int batch = 0; batch < num_batches; batch++) {
      CHECK(actual->GetPageInfo(col, batch) == std::make_tuple(col * 10 + batch, col * 10 + batch));
    }
  }
}