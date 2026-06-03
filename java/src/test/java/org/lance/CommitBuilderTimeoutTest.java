/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.lance;

import org.lance.operation.Append;
import org.lance.operation.OperationTestBase;

import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.time.Duration;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class CommitBuilderTimeoutTest extends OperationTestBase {

  @Test
  void testZeroTimeoutRejected(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("zeroTimeout").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();
      assertThrows(
          IllegalArgumentException.class,
          () -> new CommitBuilder(dataset).commitTimeout(Duration.ZERO));
      assertThrows(
          IllegalArgumentException.class,
          () -> new CommitBuilder(dataset).commitTimeout(Duration.ofMillis(-1)));
    }
  }

  @Test
  void testNullDisablesTimeout(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("nullTimeout").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();
      FragmentMetadata fragment = testDataset.createNewFragment(10);
      try (Transaction txn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(Append.builder().fragments(Collections.singletonList(fragment)).build())
              .build()) {
        try (Dataset committed = new CommitBuilder(dataset).commitTimeout(null).execute(txn)) {
          assertEquals(2, committed.version());
        }
      }
    }
  }

  @Test
  void testPositiveTimeoutSucceeds(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("posTimeout").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();
      FragmentMetadata fragment = testDataset.createNewFragment(10);
      try (Transaction txn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(Append.builder().fragments(Collections.singletonList(fragment)).build())
              .build()) {
        try (Dataset committed =
            new CommitBuilder(dataset).commitTimeout(Duration.ofMinutes(5)).execute(txn)) {
          assertEquals(2, committed.version());
        }
      }
    }
  }

  // Timeout *firing* behavior is covered by the Rust test
  // `test_commit_timeout_triggers`, which uses a throttled store for a reliable
  // trigger; reproducing it from Java without exposing throttling would be
  // flaky on fast runners.
}
