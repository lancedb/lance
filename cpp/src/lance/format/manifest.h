#pragma once

#include <arrow/io/api.h>
#include <arrow/result.h>

#include <memory>

namespace lance::format {

class Schema;

/// Manifest
///
/// Organize the less-frequently updated metadata about the full dataset:
///  * Primary key
///  * Schema
///
class Manifest final {
 public:
  Manifest() = default;

  /// Construct a Manifest with primary key and schema.
  ///
  /// \param primary_key
  /// \param schema
  ///
  Manifest(const std::string& primary_key, std::shared_ptr<Schema> schema);

  Manifest(Manifest&& other);

  ~Manifest();

  /// Parse a Manifest from input file at the offset.
  static ::arrow::Result<std::shared_ptr<Manifest>> Parse(
      std::shared_ptr<::arrow::io::RandomAccessFile> in, int64_t offset);

  /** Write Manifest to file */
  ::arrow::Result<int64_t> Write(std::shared_ptr<::arrow::io::OutputStream> out) const;

  std::string primary_key() const;

  /// Get schema.
  const Schema& schema() const;

 private:
  /// Primary key of the datasets.
  std::string primary_key_;

  /// Table schema.
  std::shared_ptr<Schema> schema_;

  int32_t num_physical_columns_;
};

}  // namespace lance::format
