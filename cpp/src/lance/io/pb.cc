#include "lance/io/pb.h"

namespace lance::io {

/// Write the protobuf message and returns the offset the message was written at.
::arrow::Result<int64_t> WriteProto(std::shared_ptr<::arrow::io::OutputStream> sink,
                                    const google::protobuf::MessageLite& pb) {
  ARROW_ASSIGN_OR_RAISE(auto offset, sink->Tell());
  int32_t pb_length = pb.ByteSizeLong();
  ARROW_RETURN_NOT_OK(WriteInt<int32_t>(sink, pb_length));
  ARROW_RETURN_NOT_OK(sink->Write(pb.SerializeAsString()));
  return offset;
}

}  // namespace lance::io
