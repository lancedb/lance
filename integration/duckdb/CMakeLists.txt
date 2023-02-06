cmake_minimum_required(VERSION 3.22)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

project(lance_duckdb VERSION 0.3)

include(FetchContent)

FetchContent_Declare(
  Corrosion
  GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
  GIT_TAG v0.3.2 # Optionally specify a commit hash, version tag or branch here
)

FetchContent_MakeAvailable(Corrosion)
