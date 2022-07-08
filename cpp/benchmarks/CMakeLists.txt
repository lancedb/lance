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

FetchContent_Declare(argparse URL https://github.com/p-ranav/argparse/archive/refs/tags/v2.5.tar.gz)
FetchContent_MakeAvailable(argparse)

find_package(Parquet REQUIRED)
include_directories(${PARQUET_INCLUDE_DIR})

add_library(lance_benchmark
        OBJECT
        bench_utils.cc
        bench_utils.h
        )
target_link_libraries(lance_benchmark ${PARQUET_SHARED_LIB})
target_include_directories(
        lance_benchmark
        SYSTEM
        PRIVATE
        ${ARROW_INCLUDE_DIR}
        ${PARQUET_INCLUDE_DIR}
)

add_executable(convert convert.cc)
target_link_libraries(
        convert
        lance
        argparse
)
target_include_directories(convert SYSTEM PRIVATE ${PARQUET_INCLUDE_DIR} ${ARROW_INCLUDE_DIR})
