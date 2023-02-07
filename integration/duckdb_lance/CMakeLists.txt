#  Copyright 2023 Lance Authors
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

# Still need to use cmake to link to duckdb via `build_loadable_extension` macro.
#

cmake_minimum_required(VERSION 3.22)

if (POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif ()

project(lance_duckdb VERSION 0.3)
set(EXTENSION_NAME lance)

if (APPLE)
  # POLICY CMP0042
  set(CMAKE_MACOSX_RPATH 1)
endif()

include(FetchContent)

FetchContent_Declare(
        Corrosion
        GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
        GIT_TAG v0.3.2 # Optionally specify a commit hash, version tag or branch here
)
set(BUILD_UNITTESTS FALSE) # Disable unit test build in duckdb

FetchContent_MakeAvailable(Corrosion)

#set(EXTERNAL_EXTENSION_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR})

corrosion_import_crate(MANIFEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/duckdb/src/include)

set(ALL_SOURCES src/extension.c src/extension.h)

set(PARAMETERS "-warnings")
build_loadable_extension(${EXTENSION_NAME} ${PARAMETERS} ${ALL_SOURCES})

set(LIB_NAME ${EXTENSION_NAME}_loadable_extension)

set_target_properties(${LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${LIB_NAME}
        "${CMAKE_CURRENT_BINARY_DIR}/libduckdb_lance.a"
)

if (APPLE)
        target_link_libraries(${LIB_NAME}
                "-framework CoreFoundation"
                "-framework Security")
endif()
