cmake_minimum_required(VERSION 3.22)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

#add_compile_options(-mf16c) # opencv

project(lance_duckdb CXX)
# Build DuckDB PyTorch extension
option(LANCE_BUILD_PYTORCH "Build with PyTorch" OFF)

set(OS_ARCH "amd64")
string(REGEX MATCH "(arm64|aarch64)" IS_ARM "${CMAKE_SYSTEM_PROCESSOR}")
if(IS_ARM)
  set(OS_ARCH "arm64")
endif()

if (APPLE)
  # POLICY CMP0042
  set(CMAKE_MACOSX_RPATH 1)

  set(CMAKE_ENABLE_EXPORTS 1)
  set(CMAKE_CXX_COMPILER_ID Clang) # DuckDB CMakeLists.txt does not allow static linking for AppleClang
endif()

include(FetchContent)

set(EXTENSION_STATIC_BUILD 1)

set(BUILD_UNITTESTS FALSE) # For duckdb
FetchContent_Declare(
  DuckDB
  GIT_REPOSITORY https://github.com/duckdb/duckdb.git
  GIT_TAG v0.6.0)
list(APPEND available_contents duckdb)

if(LANCE_BUILD_PYTORCH)
  add_compile_definitions(WITH_PYTORCH)
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
        URL https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.13.0.zip)
      list(APPEND available_contents Torch)
    else()
		# Apple Silicon
		# libtorch does not have apple silicon release yet.Temporary solution.
      file(
        DOWNLOAD
		https://files.pythonhosted.org/packages/79/b3/eaea3fc35d0466b9dae1e3f9db08467939347b3aaa53c0fd81953032db33/torch-1.13.0-cp310-none-macosx_11_0_arm64.whl
        ${CMAKE_BINARY_DIR}/torch.zip)
      file(ARCHIVE_EXTRACT INPUT ${CMAKE_BINARY_DIR}/torch.zip DESTINATION
           ${CMAKE_BINARY_DIR}/thirdparty)
      file(REMOVE ${CMAKE_BINARY_DIR}/torch.zip)

      set(torch_SOURCE_DIR ${CMAKE_BINARY_DIR}/thirdparty/torch/)
    endif()
  else()
    if(LANCE_BUILD_CUDA)
      FetchContent_Declare(
        Torch
        URL https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-1.13.0%2Bcu117.zip
      )
      enable_language(CUDA)
      find_package(CUDAToolkit)
    else()
      FetchContent_Declare(
        Torch
        URL https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.0%2Bcpu.zip
      )
    endif()
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

  include_directories(
    ${OpenCV_SOURCE_DIR}/include ${OpenCV_SOURCE_DIR}/modules/core/include
    ${OpenCV_SOURCE_DIR}/modules/imgcodecs/include
    ${OpenCV_SOURCE_DIR}/modules/imgproc/include ${CMAKE_CURRENT_BINARY_DIR})

  find_package(Torch REQUIRED)

  include_directories(${TORCH_INCLUDE_DIRS})
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Add lance core as dependency
find_package(Arrow REQUIRED)
find_package(ArrowDataset REQUIRED)
include_directories(${CMAKE_BINARY_DIR}/lance/src ../../cpp/include ../../cpp/src)
add_subdirectory(../../cpp lance)

include_directories(src)

set(LANCE_EXT_SOURCE_COMMON
    src/lance/duckdb/lance_reader.cc
    src/lance/duckdb/lance_reader.h
    src/lance/duckdb/lance-extension.cc
    src/lance/duckdb/lance.cc
    src/lance/duckdb/lance.h
    src/lance/duckdb/list_functions.cc
    src/lance/duckdb/list_functions.h
    src/lance/duckdb/vector_functions.cc
    src/lance/duckdb/vector_functions.h
    src/lance/duckdb/macros.h
    src/lance/duckdb/macros.cc
)

set(LANCE_EXT_SOURCE_ML
    src/lance/duckdb/ml/catalog.cc
    src/lance/duckdb/ml/pytorch.cc
    src/lance/duckdb/ml/functions.cc)

if(LANCE_BUILD_PYTORCH)
  set(LANCE_EXT_SOURCES
      ${LANCE_EXT_SOURCE_COMMON}
      ${LANCE_EXT_SOURCE_ML})
else()
  set(LANCE_EXT_SOURCES ${LANCE_EXT_SOURCE_COMMON})
endif()

# add_library(lance_extension STATIC ${LANCE_EXT_SOURCES})
set(PARAMETERS "-warnings")
build_loadable_extension(lance ${PARAMETERS} ${LANCE_EXT_SOURCES})
target_link_libraries(lance_loadable_extension lance ArrowDataset::arrow_dataset_shared fmt::fmt)

if(LANCE_BUILD_PYTORCH)
  target_link_libraries(lance_loadable_extension "${TORCH_LIBRARIES}"
                        opencv_imgcodecs)
  if(LANCE_BUILD_CUDA)
    set_target_properties(lance_loadable_extension PROPERTIES CUDA_ARCHITECTURES native)
  endif()
endif()

