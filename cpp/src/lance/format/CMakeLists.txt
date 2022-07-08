#  Copyright 2022 Lance Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


protobuf_generate_cpp(
        PROTO_SRCS
        PROTO_HDRS
        ${CMAKE_SOURCE_DIR}/../protos/format.proto
)

add_library(
        format
        OBJECT
        format.h
        lookup_table.cc
        lookup_table.h
        manifest.cc
        manifest.h
        metadata.cc
        metadata.h
        schema.cc
        schema.h
        ${PROTO_SRCS}
        ${PROTO_HDRS}
)
target_include_directories(format SYSTEM PRIVATE ${Protobuf_INCLUDE_DIR})

add_lance_test(lookup_table_test)
add_lance_test(metadata_test)
add_lance_test(schema_test)
