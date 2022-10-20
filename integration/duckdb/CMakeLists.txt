cmake_minimum_required(VERSION 3.22)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

project(lance_duckdb CXX)

option(LANCE_BUILD_PYTORCH "Build with PyTorch" TRUE)

set(OS_ARCH "amd64")
string(REGEX MATCH "(arm64|aarch64)" IS_ARM "${CMAKE_SYSTEM_PROCESSOR}")
if(IS_ARM)
  set(OS_ARCH "arm64")
endif()

include(FetchContent)

if(APPLE)
  set(EXTENSION_STATIC_BUILD
      0
      CACHE INTERNAL "Do not build static duckdb extension")
else()
  set(EXTENSION_STATIC_BUILD 1)
endif()

set(BUILD_UNITTESTS FALSE) # For duckdb
FetchContent_Declare(
  DuckDB
  GIT_REPOSITORY https://github.com/duckdb/duckdb.git
  GIT_TAG v0.5.1)
list(APPEND available_contents duckdb)

if(LANCE_BUILD_PYTORCH)

  if(OS_ARCH STREQUAL "amd64")
    set(CPU_BASELINE AVX2)
  endif()

  set(BUILD_LIST
      imgcodecs
      CACHE INTERNAL "Do not build static duckdb extension")
  set(BUILD_SHARED_LIBS OFF)
  set(BUILD_ZLIB OFF)
  FetchContent_Declare(
    OpenCV URL https://github.com/opencv/opencv/archive/refs/tags/4.5.5.tar.gz)
  list(APPEND available_contents OpenCV)

  if(APPLE)
    if(OS_ARCH STREQUAL "amd64")
      FetchContent_Declare(
        Torch
        URL https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.12.1.zip)
      list(APPEND available_contents Torch)
    else()
      # Apple Silicon
      # libtorch does not have apple silicon release yet. Temporary solution.
      file(
        DOWNLOAD
        https://files.pythonhosted.org/packages/5a/e8/82c14c28360dafe02877b28c70218c8b6ca8a0f2fbb0515b2abd027ca251/torch-1.12.0-cp310-none-macosx_11_0_arm64.whl
        ${CMAKE_BINARY_DIR}/torch.zip)
      file(ARCHIVE_EXTRACT INPUT ${CMAKE_BINARY_DIR}/torch.zip DESTINATION
           ${CMAKE_BINARY_DIR}/thirdparty)
      file(REMOVE ${CMAKE_BINARY_DIR}/torch.zip)

      set(torch_SOURCE_DIR ${CMAKE_BINARY_DIR}/thirdparty/torch/)
    endif()
  else()
    FetchContent_Declare(
      Torch
      URL https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcpu.zip
    )
    list(APPEND available_contents Torch)
  endif()
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
    src/lance/duckdb/ml/functions.h)

build_loadable_extension(lance ${LANCE_EXT_SOURCES})
if(APPLE)
  target_link_libraries(lance_loadable_extension duckdb)
endif()

if(LANCE_BUILD_PYTORCH)
  target_link_libraries(lance_loadable_extension "${TORCH_LIBRARIES}"
                        opencv_imgcodecs)
endif()
