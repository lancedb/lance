#include "lance/format/metadata.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Test locate chunk") {

  auto metadata = lance::format::Metadata();
  metadata.AddBatchLength(10);
  metadata.AddBatchLength(20);
  metadata.AddBatchLength(30);

  {
    auto [chunk, idx] = metadata.LocateBatch(0).ValueOrDie();
    CHECK(chunk == 0);
    CHECK(idx == 0);
  }

  {
    auto [chunk, idx] = metadata.LocateBatch(5).ValueOrDie();
    CHECK(chunk == 0);
    CHECK(idx == 5);
  }

  {
    auto [chunk, idx] = metadata.LocateBatch(10).ValueOrDie();
    CHECK(chunk == 1);
    CHECK(idx == 0);
  }

  {
    auto [chunk, idx] = metadata.LocateBatch(29).ValueOrDie();
    CHECK(chunk == 1);
    CHECK(idx == 19);
  }

  {
    auto [chunk, idx] = metadata.LocateBatch(30).ValueOrDie();
    CHECK(chunk == 2);
    CHECK(idx == 0);
  }

  {
    auto [chunk, idx] = metadata.LocateBatch(59).ValueOrDie();
    CHECK(chunk == 2);
    CHECK(idx == 29);
  }

  {
    CHECK(!metadata.LocateBatch(-1).ok());
    CHECK(!metadata.LocateBatch(60).ok());
    CHECK(!metadata.LocateBatch(65).ok());
  }
}