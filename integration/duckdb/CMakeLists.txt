cmake_minimum_required(VERSION 3.22)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

project(lance_duckdb CXX)

option(LANCE_BUILD_PYTORCH "Build with PyTorch")
if(UNIX AND NOT APPLE)
  set(LANCE_BUILD_PYTORCH TRUE)
endif()

# Default to Debug build if not specified if (NOT DEFINED CMAKE_BUILD_TYPE OR
# CMAKE_BUILD_TYPE STREQUAL "") set(CMAKE_BUILD_TYPE Debug) endif ()

include(FetchContent)

if(APPLE)
  set(EXTENSION_STATIC_BUILD
      0
      CACHE INTERNAL "Do not build static duckdb extension")
else()
  set(EXTENSION_STATIC_BUILD 1)
endif()

FetchContent_Declare(
  DuckDB
  GIT_REPOSITORY https://github.com/duckdb/duckdb.git
  GIT_TAG v0.5.1
  CMAKE_ARGS -DBUILD_UNITTESTS=FALSE)
list(APPEND available_contents duckdb)

if(LANCE_BUILD_PYTORCH)

  if(APPLE)
    message(FATAL_ERROR "PyTorch integration is only supported on Linux now")
  endif()

  set(CPU_BASELINE AVX2)
  set(BUILD_LIST
      imgcodecs
      CACHE INTERNAL "Do not build static duckdb extension")
  set(BUILD_SHARED_LIBS OFF)
  FetchContent_Declare(
    OpenCV URL https://github.com/opencv/opencv/archive/refs/tags/4.5.5.tar.gz
               CMAKE_ARGS -DBUILD_UNITTESTS=imgcodecs)
  list(APPEND available_contents OpenCV)

  FetchContent_Declare(
    Torch
    URL https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcpu.zip
  )
  list(APPEND available_contents Torch)
endif()

FetchContent_MakeAvailable(${available_contents})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${duckdb_SOURCE_DIR}/src/include)

if(LANCE_BUILD_PYTORCH)
  set(Torch_DIR "${torch_SOURCE_DIR}/share/cmake/Torch")
  set(OpenCV_DIR ${CMAKE_CURRENT_BINARY_DIR})
  include_directories(${OpenCV_INCLUDE_DIRS})

  include_directories(
    ${OpenCV_SOURCE_DIR}/include ${OpenCV_SOURCE_DIR}/modules/core/include
    ${OpenCV_SOURCE_DIR}/modules/imgcodecs/include
    ${OpenCV_SOURCE_DIR}/modules/imgproc/include ${CMAKE_CURRENT_BINARY_DIR})

  # TODO: it is CPU version for now.
  find_package(Torch REQUIRED)
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

include_directories(src)
set(LANCE_EXT_SOURCES
    src/lance/duckdb/lance-extension.cc
    src/lance/duckdb/list_functions.cc
    src/lance/duckdb/list_functions.h
    src/lance/duckdb/vector_functions.cc
    src/lance/duckdb/vector_functions.h
    src/lance/duckdb/ml/catalog.cc
    src/lance/duckdb/ml/catalog.h
    src/lance/duckdb/ml/pytorch.cc
    src/lance/duckdb/ml/pytorch.h
    src/lance/duckdb/ml/functions.cc
    src/lance/duckdb/ml/functions.h
    )

set(EXTENSION_STATIC_BUILD 1)
build_loadable_extension(lance ${LANCE_EXT_SOURCES})
if(APPLE)
  target_link_libraries(lance_loadable_extension duckdb)
endif()

if(LANCE_BUILD_PYTORCH)
  target_link_libraries(lance_loadable_extension "${TORCH_LIBRARIES}"
                        opencv_imgcodecs)
endif()
