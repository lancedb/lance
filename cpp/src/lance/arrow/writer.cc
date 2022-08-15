#include "lance/arrow/writer.h"

#include <arrow/io/api.h>
#include <arrow/record_batch.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <fmt/format.h>

#include <map>
#include <optional>
#include <vector>

#include "lance/arrow/file_lance.h"
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
  lance::io::FileWriter writer(table.schema(), opts, sink, {});

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