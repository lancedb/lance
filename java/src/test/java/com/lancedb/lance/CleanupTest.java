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
package com.lancedb.lance;

import com.lancedb.lance.cleanup.CleanupPolicy;
import com.lancedb.lance.cleanup.RemovalStats;

import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class CleanupTest {
  @Test
  public void testCleanupBeforeVersion(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("test_dataset_for_cleanup").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      testDataset.createEmptyDataset().close();

      testDataset.write(1, 10).close();
      testDataset.write(2, 10).close();

      try (Dataset dataset = testDataset.write(3, 10)) {
        RemovalStats stats =
            dataset.cleanupWithPolicy(CleanupPolicy.builder().withBeforeVersion(3L).build());
        assertEquals(2L, stats.getOldVersions());
      }
    }
  }

  @Test
  public void testCleanupBeforeTimestamp(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("test_dataset_for_cleanup").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      testDataset.createEmptyDataset().close();

      testDataset.write(1, 10).close();

      Thread.sleep(100L);
      long beforeTs = System.currentTimeMillis();

      testDataset.write(2, 10).close();

      try (Dataset dataset = testDataset.write(3, 10)) {
        RemovalStats stats =
            dataset.cleanupWithPolicy(
                CleanupPolicy.builder().withBeforeTimestampMillis(beforeTs).build());
        assertEquals(2L, stats.getOldVersions());
      }
    }
  }

  @Test
  public void testCleanupTaggedVersion(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("test_dataset_for_cleanup").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      testDataset.createEmptyDataset().close();

      Dataset ds = testDataset.write(1, 10);
      ds.tags().create("tag-2", 2L);

      testDataset.write(2, 10).close();

      try (Dataset dataset = testDataset.write(3, 10)) {
        // cleanup with tag-2 should throw exception
        Assertions.assertThrows(
            RuntimeException.class,
            () ->
                dataset.cleanupWithPolicy(
                    CleanupPolicy.builder()
                        .withErrorIfTaggedOldVersions(true)
                        .withBeforeVersion(3L)
                        .build()));

        // cleanup with tag-2 should not throw exception when set errorIfTaggedOldVersions to false
        RemovalStats stats =
            dataset.cleanupWithPolicy(
                CleanupPolicy.builder()
                    .withErrorIfTaggedOldVersions(false)
                    .withBeforeVersion(3L)
                    .build());
        assertEquals(1L, stats.getOldVersions());

        // The version with tag-2 should not be cleaned up
        Assertions.assertEquals("tag-2", dataset.tags().list().get(0).getName());
      }
    }
  }
}
