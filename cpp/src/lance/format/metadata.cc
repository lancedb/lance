#include "lance/format/metadata.h"

#include <arrow/buffer.h>
#include <arrow/result.h>
#include <fmt/format.h>

#include <memory>
#include <numeric>
#include <tuple>

#include "lance/format/format.h"
#include "lance/format/manifest.h"
#include "lance/io/pb.h"

using arrow::Result;
using arrow::Status;
using std::shared_ptr;

namespace lance::format {

Result<shared_ptr<Metadata>> Metadata::Parse(shared_ptr<::arrow::Buffer> buffer) {
  auto meta = std::make_unique<Metadata>();
  auto msg = io::ParseProto<pb::Metadata>(buffer);
  if (!msg.ok()) {
    return msg.status();
  }
  meta->pb_ = std::move(*msg);
  return meta;
}

int32_t Metadata::num_chunks() const { return pb_.chunk_offsets_size() - 1; }

int64_t Metadata::length() const { return pb_.chunk_offsets(pb_.chunk_offsets_size() - 1); }

void Metadata::AddChunkOffset(int32_t chunk_length) {
  if (pb_.chunk_offsets_size() == 0) {
    pb_.add_chunk_offsets(0);
  }
  auto last = pb_.chunk_offsets(pb_.chunk_offsets_size() - 1);
  pb_.add_chunk_offsets(last + chunk_length);
}

int32_t Metadata::GetChunkLength(int32_t chunk_id) const {
  assert(chunk_id <= pb_.chunk_offsets_size());
  return pb_.chunk_offsets(chunk_id + 1) - pb_.chunk_offsets(chunk_id);
}

::arrow::Result<std::tuple<int32_t, int32_t>> Metadata::LocateChunk(int32_t idx) const {
  int64_t len = length();
  if (idx < 0 || idx >= len) {
    return ::arrow::Status::IndexError(fmt::format("Chunk index out of range: {} of {}", idx, len));
  }
  auto it = std::upper_bound(pb_.chunk_offsets().begin(), pb_.chunk_offsets().end(), idx);
  if (it == pb_.chunk_offsets().end()) {
    return ::arrow::Status::IndexError("Chunk index out of range {} of {}", idx, len);
  }
  int32_t bound_idx = std::distance(pb_.chunk_offsets().begin(), it);
  assert(bound_idx >= 0);
  bound_idx = std::max(0, bound_idx - 1);
  int32_t idx_in_chunk = idx - pb_.chunk_offsets(bound_idx);
  return std::tuple(bound_idx, idx_in_chunk);
}

::arrow::Result<std::shared_ptr<Manifest>> Metadata::GetManifest(
    std::shared_ptr<::arrow::io::RandomAccessFile> in) {
  // TODO: change to read buffer instead of read file again.
  if (pb_.manifest_position() == 0) {
    return Status::IOError("Can not find manifest within the file");
  }
  return Manifest::Parse(in, pb_.manifest_position());
}

void Metadata::SetChunkPosition(int64_t position) { pb_.set_chunk_position(position); }

}  // namespace lance::format