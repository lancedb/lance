#include "lance/format/manifest.h"

#include <arrow/result.h>

#include <memory>

#include "lance/arrow/type.h"
#include "lance/format/schema.h"
#include "lance/io/pb.h"

using arrow::Result;
using arrow::Status;

namespace lance::format {

Manifest::Manifest(std::shared_ptr<Schema> schema)
    : schema_(std::move(schema)) {}

Manifest::Manifest(Manifest&& other) noexcept
    : schema_(std::move(other.schema_)) {}

::arrow::Result<std::shared_ptr<Manifest>> Manifest::Parse(
    std::shared_ptr<::arrow::io::RandomAccessFile> in, int64_t offset) {
  ARROW_ASSIGN_OR_RAISE(auto pb, io::ParseProto<pb::Manifest>(in, offset));
  auto schema = std::make_unique<Schema>(pb.fields());
  return std::make_shared<Manifest>(std::move(schema));
}

::arrow::Result<int64_t> Manifest::Write(std::shared_ptr<::arrow::io::OutputStream> out) const {
  lance::format::pb::Manifest pb;
  for (auto field : schema_->ToProto()) {
    auto pb_field = pb.add_fields();
    *pb_field = field;
  }
  return io::WriteProto(out, pb);
}

const Schema& Manifest::schema() const { return *schema_; }

}  // namespace lance::format
