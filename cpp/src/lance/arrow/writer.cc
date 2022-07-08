#include "lance/arrow/writer.h"

#include <arrow/io/api.h>
#include <arrow/record_batch.h>
#include <arrow/status.h>
#include <arrow/table.h>

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
                           const string& primary_key,
                           std::optional<FileWriteOptions> options) {
  auto opts = std::make_shared<lance::arrow::FileWriteOptions>();
  if (options.has_value()) {
    *opts = options.value();
  }
  opts->primary_key = primary_key;
  lance::io::FileWriter writer(table.schema(), opts, sink, {});

  std::shared_ptr<::arrow::RecordBatch> batch;
  ::arrow::TableBatchReader batch_reader(table);
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