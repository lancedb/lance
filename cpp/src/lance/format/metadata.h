#pragma once

#include <arrow/io/api.h>
#include <arrow/result.h>
#include <arrow/type.h>

#include <map>
#include <memory>
#include <tuple>
#include <vector>

#include "lance/format/format.pb.h"

namespace lance::format {

class Manifest;

/** File-level Metadata. */
class Metadata final {
 public:
  Metadata() = default;

  virtual ~Metadata() = default;

  /** Parse a Metadata from an arrow buffer. */
  static ::arrow::Result<std::shared_ptr<Metadata>> Parse(std::shared_ptr<::arrow::Buffer> buffer);

  void AddChunkOffset(int32_t chunk_length);

  int32_t num_chunks() const;

  /// Get the logical length of a chunk.
  int32_t GetChunkLength(int32_t chunk_id) const;

  /// Locate the chunk index where the idx belongs.
  ///
  /// \param idx the absolute index of a row in the file.
  /// \return a tuple of <chunk id, idx in the chunk>
  ::arrow::Result<std::tuple<int32_t, int32_t>> LocateChunk(int32_t idx) const;

  int64_t length() const;

  int64_t chunk_position() const { return pb_.chunk_position(); }

  void SetChunkPosition(int64_t position);

  // Leak abstraction in prototype
  inline pb::Metadata& pb() { return pb_; }

  // THIS METHOD SUCKS.
  ::arrow::Result<std::shared_ptr<Manifest>> GetManifest(
      std::shared_ptr<::arrow::io::RandomAccessFile> in);

 private:
  pb::Metadata pb_;
};

}  // namespace lance::format
