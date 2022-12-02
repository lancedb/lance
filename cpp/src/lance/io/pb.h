/// Support SerDe protobuf to files

#pragma once

#include <arrow/buffer.h>
#include <arrow/result.h>
#include <google/protobuf/message_lite.h>

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
    return ::arrow::Status::Invalid("Failed to parse protobuf");
  };
  return proto;
}

::arrow::Result<int64_t> WriteProto(const std::shared_ptr<::arrow::io::OutputStream>& sink,
                                    const google::protobuf::Message& pb);

}  // namespace lance::io
