#pragma once

#include <arrow/type_fwd.h>

#include <future>
#include <memory>
#include <optional>
#include <queue>

// Forward declarations.
namespace arrow::dataset {
class ScanOptions;
}
namespace lance::format {
class Schema;
}

namespace lance::io {

class FileReader;

/// lance Scanner
class Scanner {
 public:
  Scanner(std::shared_ptr<FileReader> reader,
          std::shared_ptr<::arrow::dataset::ScanOptions> options);

  /// Move constructor.
  Scanner(Scanner&& other) noexcept;

  Scanner(const Scanner&);

  ~Scanner() = default;

  ::arrow::Status Open();

  ::arrow::Result<::std::shared_ptr<::arrow::RecordBatch>> Next();

  /// Async read, to match lanceFileFormat::ScanBatchesAsync()
  ::arrow::Future<std::shared_ptr<::arrow::RecordBatch>> operator()();

 private:
  Scanner() = delete;

  std::shared_ptr<FileReader> reader_;
  std::shared_ptr<::arrow::dataset::ScanOptions> options_;
  std::shared_ptr<lance::format::Schema> schema_;

  int current_offset_ = 0;
  int prefetch_offset_ = 0;
  std::queue<
      std::future<std::tuple<::arrow::Result<std::shared_ptr<::arrow::RecordBatch>>, int32_t>>>
      q_;

  void AddPrefetchTask();
};

}  // namespace lance::io
