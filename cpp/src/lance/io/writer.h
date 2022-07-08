#pragma once

#include <arrow/dataset/file_base.h>
#include <arrow/filesystem/api.h>
#include <arrow/io/api.h>
#include <arrow/status.h>
#include <arrow/util/future.h>

#include <memory>

#include "lance/format/lookup_table.h"
#include "lance/format/metadata.h"

namespace lance::format {
class Field;
class Schema;
class Metadata;
}  // namespace lance::format

namespace lance::io {

/// lance FileWriter
class FileWriter final : public ::arrow::dataset::FileWriter {
 public:
  FileWriter(std::shared_ptr<::arrow::Schema> schema,
             std::shared_ptr<::arrow::dataset::FileWriteOptions> options,
             std::shared_ptr<::arrow::io::OutputStream> destination,
             ::arrow::fs::FileLocator destination_locator);

  ~FileWriter();

  ::arrow::Status Write(const std::shared_ptr<::arrow::RecordBatch>& batch) override;

 private:
  ::arrow::Future<> FinishInternal() override;

  ::arrow::Status WriteFooter();

  ::arrow::Status WriteArray(const std::shared_ptr<format::Field>& field,
                             const std::shared_ptr<::arrow::Array>& arr);
  ::arrow::Status WritePrimitiveArray(const std::shared_ptr<format::Field>& field,
                                      const std::shared_ptr<::arrow::Array>& arr);
  ::arrow::Status WriteStructArray(const std::shared_ptr<format::Field>& field,
                                   const std::shared_ptr<::arrow::Array>& arr);
  ::arrow::Status WriteListArray(const std::shared_ptr<format::Field>& field,
                                 const std::shared_ptr<::arrow::Array>& arr);

  std::shared_ptr<lance::format::Schema> lance_schema_;
  std::unique_ptr<lance::format::Metadata> metadata_;
  format::LookupTable lookup_table_;
  int32_t chunk_id_ = 0;
};

}  // namespace lance::io