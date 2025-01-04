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

import com.lancedb.lance.ipc.LanceScanner;

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class FragmentTest {
  @TempDir private static Path tempDir; // Temporary directory for the tests

  @Test
  void testFragmentCreateFfiArray() {
    String datasetPath = tempDir.resolve("new_fragment_array").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      testDataset.createNewFragment(20);
    }
  }

  @Test
  void testFragmentCreate() throws Exception {
    String datasetPath = tempDir.resolve("new_fragment").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int rowCount = 21;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(rowCount);

      // Commit fragment
      FragmentOperation.Append appendOp = new FragmentOperation.Append(Arrays.asList(fragmentMeta));
      try (Dataset dataset = Dataset.commit(allocator, datasetPath, appendOp, Optional.of(1L))) {
        assertEquals(2, dataset.version());
        assertEquals(2, dataset.latestVersion());
        assertEquals(rowCount, dataset.countRows());
        Fragment fragment = dataset.getFragments().get(0);

        try (LanceScanner scanner = fragment.newScan()) {
          Schema schemaRes = scanner.schema();
          assertEquals(testDataset.getSchema(), schemaRes);
        }
      }
    }
  }

  @Test
  void commitWithoutVersion() {
    String datasetPath = tempDir.resolve("commit_without_version").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      FragmentMetadata meta = testDataset.createNewFragment(20);
      FragmentOperation.Append appendOp = new FragmentOperation.Append(Arrays.asList(meta));
      assertThrows(
          IllegalArgumentException.class,
          () -> {
            Dataset.commit(allocator, datasetPath, appendOp, Optional.empty());
          });
    }
  }

  @Test
  void commitOldVersion() {
    String datasetPath = tempDir.resolve("commit_old_version").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      FragmentMetadata meta = testDataset.createNewFragment(20);
      FragmentOperation.Append appendOp = new FragmentOperation.Append(Arrays.asList(meta));
      assertThrows(
          IllegalArgumentException.class,
          () -> {
            Dataset.commit(allocator, datasetPath, appendOp, Optional.of(0L));
          });
    }
  }

  @Test
  void appendWithoutFragment() {
    String datasetPath = tempDir.resolve("append_without_fragment").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      assertThrows(
          IllegalArgumentException.class,
          () -> {
            new FragmentOperation.Append(new ArrayList<>());
          });
    }
  }

  @Test
  void testOverwriteCommit() throws Exception {
    String datasetPath = tempDir.resolve("testOverwriteCommit").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();

      // Commit fragment
      int rowCount = 20;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(rowCount);
      FragmentOperation.Overwrite overwrite =
          new FragmentOperation.Overwrite(
              Collections.singletonList(fragmentMeta), testDataset.getSchema());
      try (Dataset dataset = Dataset.commit(allocator, datasetPath, overwrite, Optional.of(1L))) {
        assertEquals(2, dataset.version());
        assertEquals(2, dataset.latestVersion());
        assertEquals(rowCount, dataset.countRows());
        Fragment fragment = dataset.getFragments().get(0);

        try (LanceScanner scanner = fragment.newScan()) {
          Schema schemaRes = scanner.schema();
          assertEquals(testDataset.getSchema(), schemaRes);
        }
      }

      // Commit fragment again
      rowCount = 40;
      fragmentMeta = testDataset.createNewFragment(rowCount);
      overwrite =
          new FragmentOperation.Overwrite(
              Collections.singletonList(fragmentMeta), testDataset.getSchema());
      try (Dataset dataset = Dataset.commit(allocator, datasetPath, overwrite, Optional.of(2L))) {
        assertEquals(3, dataset.version());
        assertEquals(3, dataset.latestVersion());
        assertEquals(rowCount, dataset.countRows());
        Fragment fragment = dataset.getFragments().get(0);

        try (LanceScanner scanner = fragment.newScan()) {
          Schema schemaRes = scanner.schema();
          assertEquals(testDataset.getSchema(), schemaRes);
        }
      }
    }
  }

  @Test
  void testEmptyFragments() {
    String datasetPath = tempDir.resolve("testEmptyFragments").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      List<FragmentMetadata> fragments = testDataset.createNewFragment(0, 10);
      assertEquals(0, fragments.size());
    }
  }

  @Test
  void testMultiFragments() {
    String datasetPath = tempDir.resolve("testMultiFragments").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      List<FragmentMetadata> fragments = testDataset.createNewFragment(20, 10);
      assertEquals(2, fragments.size());
    }
  }
}
