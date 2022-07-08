#pragma once

#include <arrow/io/api.h>
#include <arrow/result.h>

#include <map>
#include <memory>
#include <vector>

namespace lance::format {

namespace pb {
class Metadata;
}

class LookupTable {
 public:
  LookupTable() = default;

  void AddOffset(int32_t column, int32_t chunk, int64_t offset);

  void AddPageLength(int32_t column, int32_t chunk, int64_t length);

  int64_t GetOffset(int32_t column_id, int32_t chunk_id) const;

  ::arrow::Result<int64_t> GetPageLength(int32_t column_id, int32_t chunk_id) const;

  ::arrow::Result<int64_t> Write(std::shared_ptr<::arrow::io::OutputStream> out);

  void WritePageLengthTo(pb::Metadata* out);

  /// Read lookup table from the opened file.
  ///
  static ::arrow::Result<std::shared_ptr<LookupTable>> Read(
      const std::shared_ptr<::arrow::io::RandomAccessFile>& in,
      int64_t offset,
      const pb::Metadata& pb);

 private:
  /// Map<column, Map<chunk, offset>>
  std::map<int32_t, std::map<int32_t, int64_t>> offsets_;

  /// Map<column, Map<chunk, length>>
  std::map<int32_t, std::map<int32_t, int64_t>> lengths_;

  std::vector<int64_t> page_lengths_;
};

}  // namespace lance::format
