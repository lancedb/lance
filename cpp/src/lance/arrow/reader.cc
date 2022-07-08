#include "lance/arrow/reader.h"

#include <arrow/buffer.h>
#include <arrow/io/api.h>
#include <arrow/memory_pool.h>
#include <arrow/result.h>
#include <arrow/status.h>

#include <memory>
#include <stdexcept>

#include "lance/encodings/binary.h"
#include "lance/format/manifest.h"
#include "lance/format/metadata.h"
#include "lance/format/schema.h"
#include "lance/io/reader.h"

using arrow::Result;
using arrow::Status;
using std::string;
using std::unique_ptr;

namespace pb = lance::format::pb;

namespace lance::arrow {

//------- FileReader implementation.

class FileReader::Impl {
 public:
  Impl(std::shared_ptr<::arrow::io::RandomAccessFile> infile, ::arrow::MemoryPool* pool) noexcept;

  const std::unique_ptr<lance::io::FileReader>& reader() const { return reader_; }

  std::unique_ptr<lance::io::FileReader>& reader() { return reader_; }

 private:
  Impl() = delete;

  std::unique_ptr<lance::io::FileReader> reader_;
};

FileReader::Impl::Impl(std::shared_ptr<::arrow::io::RandomAccessFile> infile,
                       ::arrow::MemoryPool* pool) noexcept
    : reader_(std::make_unique<lance::io::FileReader>(infile, pool)) {}

FileReader::FileReader(std::shared_ptr<::arrow::io::RandomAccessFile> in,
                       ::arrow::MemoryPool* pool) noexcept
    : impl_(std::make_unique<FileReader::Impl>(in, pool)) {}

FileReader::~FileReader() {}

Result<unique_ptr<FileReader>> FileReader::Make(std::shared_ptr<::arrow::io::RandomAccessFile> in,
                                                ::arrow::MemoryPool* pool) {
  auto reader = unique_ptr<FileReader>(new FileReader(in, pool));
  ARROW_RETURN_NOT_OK(reader->impl_->reader()->Open());
  return reader;
}

string FileReader::primary_key() const { return impl_->reader()->manifest().primary_key(); }

int64_t FileReader::num_chunks() const { return impl_->reader()->metadata().num_chunks(); }

int64_t FileReader::length() const { return impl_->reader()->metadata().length(); }

::arrow::Result<std::shared_ptr<::arrow::Schema>> FileReader::GetSchema() {
  return impl_->reader()->schema().ToArrow();
}

::arrow::Result<std::shared_ptr<::arrow::Table>> FileReader::ReadTable() {
  return impl_->reader()->ReadTable();
}

::arrow::Result<std::shared_ptr<::arrow::Table>> FileReader::ReadTable(
    const std::vector<std::string>& columns) {
  return impl_->reader()->ReadTable(columns);
}

::arrow::Result<std::vector<std::shared_ptr<::arrow::Scalar>>> FileReader::Get(int32_t idx) {
  return impl_->reader()->Get(idx);
}

::arrow::Result<std::vector<std::shared_ptr<::arrow::Scalar>>> FileReader::Get(
    int32_t idx, const std::vector<std::string>& columns) {
  return impl_->reader()->Get(idx, columns);
}

}  // namespace lance::arrow
