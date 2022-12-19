/// Support SerDe protobuf to files

#pragma once

#include <arrow/buffer.h>
#include <arrow/result.h>
#include <fmt/format.h>
#include <google/protobuf/message_lite.h>
#include <google/protobuf/util/time_util.h>

#include <chrono>
#include <cstring>
#include <type_traits>

#include "lance/io/endian.h"

namespace lance::io {

template <typename T>
concept ProtoMessage = std::is_base_of<google::protobuf::Message, T>::value;

template <ProtoMessage P>
::arrow::Result<P> ParseProto(const std::shared_ptr<::arrow::Buffer>& buf) {
  auto pb_size = ReadInt<int32_t>(*buf);
  P proto;
  if (!proto.ParseFromArray(buf->data() + sizeof(pb_size), pb_size)) {
    return ::arrow::Status::Invalid("Failed to parse protobuf");
  }
  return proto;
}

template <ProtoMessage P>
::arrow::Result<P> ParseProto(const std::shared_ptr<::arrow::io::RandomAccessFile>& source,
                              int64_t offset) {
  ARROW_ASSIGN_OR_RAISE(auto pb_size, ReadInt<int32_t>(source, offset));
  P proto;
  ARROW_ASSIGN_OR_RAISE(auto buf, source->ReadAt(offset + sizeof(pb_size), pb_size));
  if (!proto.ParseFromArray(buf->data(), buf->size())) {
    return ::arrow::Status::Invalid(fmt::format(
        "Failed to parse protobuf at offset {}: expected protobuf size={} buffer size={}",
        offset,
        pb_size,
        buf->size()));
  }
  return proto;
}

::arrow::Result<int64_t> WriteProto(const std::shared_ptr<::arrow::io::OutputStream>& sink,
                                    const google::protobuf::Message& pb);

/// Convert a Protobuf Timestamp to Chrono time point, in microseconds resolution.
std::chrono::time_point<std::chrono::system_clock> FromProto(
    const google::protobuf::Timestamp& proto);

/// Convert chrono time_point to protobuf Timestamp, in microseconds resolution.
google::protobuf::Timestamp ToProto(const std::chrono::time_point<std::chrono::system_clock>& ts);

}  // namespace lance::io
