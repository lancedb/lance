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

/// Lance Scanner
class Scanner {
 public:
  /// Constructor.
  Scanner(std::shared_ptr<FileReader> reader,
          std::shared_ptr<::arrow::dataset::ScanOptions> options);

  /// Move constructor.
  Scanner(Scanner&& other) noexcept;

  /// Copy constructor.
  Scanner(const Scanner&);

  ~Scanner() = default;

  /// Open the Scanner. Must call it before start iterating.
  ::arrow::Status Open();

  /// Returns the next record batch if any.
  ///
  /// \return A record batch. Returns `nullptr` if reaches the end.
  ::arrow::Result<::std::shared_ptr<::arrow::RecordBatch>> Next();

  /// Async read, to match LanceFileFormat::ScanBatchesAsync()
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
