//  Copyright 2022 Lance Authors
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "lance/arrow/writer.h"

#include <arrow/io/api.h>
#include <arrow/record_batch.h>
#include <arrow/status.h>
#include <arrow/table.h>

#include <map>
#include <vector>

#include "lance/arrow/file_lance.h"
#include "lance/format/schema.h"
#include "lance/io/writer.h"

using arrow::Status;
using arrow::Table;
using std::map;
using std::shared_ptr;
using std::string;
using std::vector;

namespace lance::arrow {

::arrow::Status WriteTable(const ::arrow::Table& table,
                           shared_ptr<::arrow::io::OutputStream> sink,
                           FileWriteOptions options) {
  ARROW_RETURN_NOT_OK(options.Validate());
  auto opts = std::make_shared<lance::arrow::FileWriteOptions>(options);
  auto schema = std::make_shared<lance::format::Schema>(table.schema());
  lance::io::FileWriter writer(schema, opts, sink, {});

  std::shared_ptr<::arrow::RecordBatch> batch;
  ::arrow::TableBatchReader batch_reader(table);
  batch_reader.set_chunksize(options.batch_size);
  while (true) {
    ARROW_RETURN_NOT_OK(batch_reader.ReadNext(&batch));
    if (!batch) {
      break;
    }
    ARROW_RETURN_NOT_OK(writer.Write(batch));
  }
  writer.Finish().Wait();
  return ::arrow::Status::OK();
}

}  // namespace lance::arrow