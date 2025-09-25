In 0.36.0 we attempted to move the index details message from table.proto into
index.proto.  However, changing the package namespace of the protobuf messages
in this way caused a forwards compatibility problem and we reverted this change.

As a result, any indexes created by version 0.36.0 will have an odd protobuf
package and we should make sure we handle this in future versions.
