#include "lance/format/manifest.h"

#include <arrow/result.h>

#include <memory>

#include "lance/arrow/type.h"
#include "lance/format/format.h"
#include "lance/format/schema.h"
#include "lance/io/pb.h"

using arrow::Result;
using arrow::Status;

namespace lance::format {

Manifest::Manifest(const std::string& primary_key, std::shared_ptr<Schema> schema)
    : primary_key_(primary_key), schema_(std::move(schema)) {
  auto protos = schema_->ToProto();
  int max_field_id = 0;
  for (auto field : schema_->ToProto()) {
    max_field_id = std::max(max_field_id, field.id());
  }
  num_physical_columns_ = max_field_id + 1;
}

Manifest::Manifest(Manifest&& other)
    : primary_key_(other.primary_key_),
      schema_(std::move(other.schema_)),
      num_physical_columns_(other.num_physical_columns_) {}

Manifest::~Manifest() {}

::arrow::Result<std::shared_ptr<Manifest>> Manifest::Parse(
    std::shared_ptr<::arrow::io::RandomAccessFile> in, int64_t offset) {
  ARROW_ASSIGN_OR_RAISE(auto pb, io::ParseProto<pb::Manifest>(in, offset));
  auto schema = std::make_unique<Schema>(pb.fields());
  return std::make_shared<Manifest>(pb.primary_key(), std::move(schema));
}

::arrow::Result<int64_t> Manifest::Write(std::shared_ptr<::arrow::io::OutputStream> out) const {
  lance::format::pb::Manifest pb;
  pb.set_primary_key(primary_key_);
  for (auto field : schema_->ToProto()) {
    auto pb_field = pb.add_fields();
    *pb_field = field;
  }
  return io::WriteProto(out, pb);
}

std::string Manifest::primary_key() const { return primary_key_; }

const Schema& Manifest::schema() const { return *schema_; }

}  // namespace lance::format
