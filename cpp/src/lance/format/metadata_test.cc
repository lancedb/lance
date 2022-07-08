#include "lance/format/metadata.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Test locate chunk") {

  auto metadata = lance::format::Metadata();
  metadata.AddChunkOffset(10);
  metadata.AddChunkOffset(20);
  metadata.AddChunkOffset(30);

  {
    auto [chunk, idx] = metadata.LocateChunk(0).ValueOrDie();
    CHECK(chunk == 0);
    CHECK(idx == 0);
  }

  {
    auto [chunk, idx] = metadata.LocateChunk(5).ValueOrDie();
    CHECK(chunk == 0);
    CHECK(idx == 5);
  }

  {
    auto [chunk, idx] = metadata.LocateChunk(10).ValueOrDie();
    CHECK(chunk == 1);
    CHECK(idx == 0);
  }

  {
    auto [chunk, idx] = metadata.LocateChunk(29).ValueOrDie();
    CHECK(chunk == 1);
    CHECK(idx == 19);
  }

  {
    auto [chunk, idx] = metadata.LocateChunk(30).ValueOrDie();
    CHECK(chunk == 2);
    CHECK(idx == 0);
  }

  {
    auto [chunk, idx] = metadata.LocateChunk(59).ValueOrDie();
    CHECK(chunk == 2);
    CHECK(idx == 29);
  }

  {
    CHECK(!metadata.LocateChunk(-1).ok());
    CHECK(!metadata.LocateChunk(60).ok());
    CHECK(!metadata.LocateChunk(65).ok());
  }
}