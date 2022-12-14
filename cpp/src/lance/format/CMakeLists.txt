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

add_custom_target(gen_protos
    COMMAND protoc -I${PROJECT_SOURCE_DIR}/../protos
    -I${protobuf_SOURCE_DIR}/src
    --cpp_out=${PROJECT_BINARY_DIR}/src/lance/format
    format.proto
    DEPENDS libprotobuf)

add_library(
    format
    OBJECT
    ${PROJECT_BINARY_DIR}/src/lance/format/format.pb.cc
    ${PROJECT_BINARY_DIR}/src/lance/format/format.pb.h
    data_fragment.cc
    data_fragment.h
    format.h
    manifest.cc
    manifest.h
    metadata.cc
    metadata.h
    page_table.cc
    page_table.h
    schema.cc
    schema.h
    visitors.cc
    visitors.h
)
add_dependencies(format gen_protos)
target_include_directories(format SYSTEM PRIVATE ${Protobuf_INCLUDE_DIR})

add_lance_test(metadata_test)
add_lance_test(page_table_test)
add_lance_test(schema_test)
