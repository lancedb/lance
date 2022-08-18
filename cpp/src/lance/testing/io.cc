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

#include "lance/testing/io.h"

#include <arrow/io/api.h>
#include <arrow/result.h>

#include "lance/arrow/writer.h"
#include "lance/io/reader.h"

namespace lance::testing {

::arrow::Result<std::shared_ptr<io::FileReader>> MakeReader(
    const std::shared_ptr<::arrow::Table>& table) {
  auto sink = ::arrow::io::BufferOutputStream::Create().ValueOrDie();
  ARROW_RETURN_NOT_OK(lance::arrow::WriteTable(*table, sink));
  auto infile = make_shared<::arrow::io::BufferReader>(sink->Finish().ValueOrDie());
  auto reader = std::make_shared<io::FileReader>(infile);
  ARROW_RETURN_NOT_OK(reader->Open());
  return reader;
}

}  // namespace lance::testing
