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

#include <arrow/io/api.h>

#include <catch2/catch_test_macros.hpp>

#include "lance/format/format.pb.h"
#include "lance/format/page_table.h"

using lance::format::PageTable;

TEST_CASE("Serialize Chunk length") {
  lance::format::PageTable lt;

  int num_columns = 3;
  int num_chunks = 5;
  for (int col = 0; col < num_columns; col++) {
    for (int chk = 0; chk < num_chunks; chk++) {
      lt.AddOffset(col, chk, col * 10 + chk);
      lt.AddPageLength(col, chk, col * 10 + chk);
    }
  }
  lance::format::pb::Metadata metadata;

  auto out_buf = arrow::io::BufferOutputStream::Create().ValueOrDie();
  lt.Write(out_buf).ValueOrDie();
  lt.WritePageLengthTo(&metadata);

  auto in_buf = std::make_shared<arrow::io::BufferReader>(out_buf->Finish().ValueOrDie());
  auto actual = PageTable::Read(in_buf, 0, metadata).ValueOrDie();

  for (int col = 0; col < num_columns; col++) {
    for (int chk = 0; chk < num_chunks; chk++) {
      INFO("Getting offset col=" << col << " chk=" << chk);
      CHECK(actual->GetOffset(col, chk) == col * 10 + chk);
      CHECK(actual->GetPageLength(col, chk).ValueOrDie() == col * 10 + chk);
    }
  }
}