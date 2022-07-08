#include "lance/format/format.h"

#include <arrow/buffer.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/type.h>
#include <fmt/format.h>

#include <cstring>
#include <string>

#include "lance/format/format.pb.h"

using arrow::Buffer;
using arrow::Result;
using arrow::Status;

using std::shared_ptr;
using std::string;

namespace lance::format {

Result<pb::DataType> ArrowTypeToPhysicalType(std::shared_ptr<::arrow::DataType> t) {
  switch (t->id()) {
    case ::arrow::Type::BOOL:
      return pb::DataType::BOOLEAN;
    case ::arrow::Type::INT32:
      return pb::DataType::INT32;
    case ::arrow::Type::INT64:
      return pb::DataType::INT64;
    case ::arrow::Type::FLOAT:
      return pb::DataType::FLOAT32;
    case ::arrow::Type::DOUBLE:
      return pb::DataType::FLOAT64;
    case ::arrow::Type::STRING:
    case ::arrow::Type::LARGE_STRING:
    case ::arrow::Type::BINARY:
    case ::arrow::Type::LARGE_BINARY:
      return pb::DataType::BYTES;
    default:
      return Status::NotImplemented(fmt::format("Unsupported type: {}", t->ToString()));
  }
}

}  // namespace lance::format