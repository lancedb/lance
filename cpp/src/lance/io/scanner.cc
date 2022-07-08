#include "lance/io/scanner.h"

#include <arrow/dataset/scanner.h>
#include <arrow/record_batch.h>
#include <arrow/util/future.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <future>
#include <set>
#include <tuple>

#include "lance/format/metadata.h"
#include "lance/format/schema.h"
#include "lance/io/reader.h"

namespace lance::io {

Scanner::Scanner(std::shared_ptr<FileReader> reader,
                 std::shared_ptr<arrow::dataset::ScanOptions> options)
    : reader_(reader), options_(options) {}

Scanner::Scanner(const Scanner& other)
    : reader_(other.reader_),
      options_(other.options_),
      schema_(other.schema_),
      current_offset_(other.current_offset_),
      prefetch_offset_(other.prefetch_offset_) {}

Scanner::Scanner(Scanner&& other) noexcept
    : reader_(std::move(other.reader_)),
      options_(std::move(other.options_)),
      schema_(std::move(other.schema_)),
      current_offset_(other.current_offset_),
      prefetch_offset_(other.prefetch_offset_),
      q_(std::move(other.q_)) {}

::arrow::Status Scanner::Open() {
  schema_ = std::make_shared<lance::format::Schema>(reader_->schema());
  std::set<std::string> columns;
  for (auto& ref : options_->MaterializedFields()) {
    // TODO: support nested columns later.
    columns.insert(*ref.name());
    auto fields = ref.FindAll(*options_->dataset_schema);
  }
  /// TODO Make schema->Project takes generic container.
  std::vector<std::string> column_vector(columns.begin(), columns.end());
  if (!columns.empty()) {
    ARROW_ASSIGN_OR_RAISE(schema_, schema_->Project(column_vector));
  }
  return ::arrow::Status::OK();
}

void Scanner::AddPrefetchTask() {
  while (q_.size() < static_cast<std::size_t>(options_->batch_readahead) &&
         prefetch_offset_ < reader_->metadata().length()) {
    auto start = prefetch_offset_;
    auto length = static_cast<int32_t>(
        std::min(static_cast<int64_t>(options_->batch_size), reader_->metadata().length() - start));
    auto f = std::async(
        [&](int32_t start, int32_t length) {
          auto batch = reader_->ReadBatch(start, length, *schema_);
          if (!batch.ok()) {
            fmt::print(
                "Bad batch: start={}, length={}: {}\n", start, length, batch.status().message());
          }
          return std::make_tuple(batch, length);
        },
        start,
        length);
    q_.push(std::move(f));
    prefetch_offset_ += length;
  }
}

::arrow::Result<::std::shared_ptr<::arrow::RecordBatch>> Scanner::Next() {
  // Let's do something simple.
  // Each time just read batch_size * prefecth_size, to amortize I/O.
  AddPrefetchTask();
  if (q_.empty()) {
    return nullptr;
  }
  auto future = std::move(q_.front());
  q_.pop();
  auto [result, length] = future.get();
  if (!result.ok()) {
    current_offset_ += options_->batch_size;
    return result.status();
  }
  auto batch = *result;
  current_offset_ += (*result)->num_rows();
  return batch;
}

::arrow::Future<std::shared_ptr<::arrow::RecordBatch>> Scanner::operator()() {
  /// TODO: Make it truly async someday.
  auto f = ::arrow::Future<std::shared_ptr<::arrow::RecordBatch>>::MakeFinished(this->Next());
  return f;
}

}  // namespace lance::io