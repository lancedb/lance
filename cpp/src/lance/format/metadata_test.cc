//  Copyright 2022 Lance Authors
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "lance/format/metadata.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Test locate batch") {
  auto metadata = lance::format::Metadata();
  metadata.AddBatchLength(10);
  metadata.AddBatchLength(20);
  metadata.AddBatchLength(30);

  {
    auto [batch_id, idx] = metadata.LocateBatch(0).ValueOrDie();
    CHECK(batch_id == 0);
    CHECK(idx == 0);
  }

  {
    auto [batch_id, idx] = metadata.LocateBatch(5).ValueOrDie();
    CHECK(batch_id == 0);
    CHECK(idx == 5);
  }

  {
    auto [batch_id, idx] = metadata.LocateBatch(10).ValueOrDie();
    CHECK(batch_id == 1);
    CHECK(idx == 0);
  }

  {
    auto [batch_id, idx] = metadata.LocateBatch(29).ValueOrDie();
    CHECK(batch_id == 1);
    CHECK(idx == 19);
  }

  {
    auto [batch_id, idx] = metadata.LocateBatch(30).ValueOrDie();
    CHECK(batch_id == 2);
    CHECK(idx == 0);
  }

  {
    auto [batch_id, idx] = metadata.LocateBatch(59).ValueOrDie();
    CHECK(batch_id == 2);
    CHECK(idx == 29);
  }

  {
    CHECK(!metadata.LocateBatch(-1).ok());
    CHECK(!metadata.LocateBatch(60).ok());
    CHECK(!metadata.LocateBatch(65).ok());
  }
}