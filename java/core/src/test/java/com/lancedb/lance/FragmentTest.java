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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.lancedb.lance.ipc.LanceScanner;
import com.lancedb.lance.ipc.ScanOptions;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.apache.arrow.dataset.scanner.Scanner;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

public class FragmentTest {
  @TempDir private static Path tempDir; // Temporary directory for the tests

  @Test
  void testFragmentCreateFfiArray() {
    String datasetPath = tempDir.resolve("new_fragment_array").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      testDataset.createNewFragment(123, 20);
    }
  }

  @Test
  void testFragmentCreate() throws Exception {
    String datasetPath = tempDir.resolve("new_fragment").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int fragmentId = 312;
      int rowCount = 21;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(fragmentId, rowCount);

      // Commit fragment
      FragmentOperation.Append appendOp = new FragmentOperation.Append(Arrays.asList(fragmentMeta));
      try (Dataset dataset = Dataset.commit(allocator, datasetPath, appendOp, Optional.of(1L))) {
        assertEquals(2, dataset.version());
        assertEquals(2, dataset.latestVersion());
        assertEquals(rowCount, dataset.countRows());
        DatasetFragment fragment = dataset.getFragments().get(0);
        assertEquals(fragmentId, fragment.getId());
  
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
      TestUtils.SimpleTestDataset testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      FragmentMetadata meta = testDataset.createNewFragment(123, 20);
      FragmentOperation.Append appendOp = new FragmentOperation.Append(Arrays.asList(meta));
      assertThrows(IllegalArgumentException.class, () -> {
        Dataset.commit(allocator, datasetPath, appendOp, Optional.empty());
      });
    }
  }

  @Test
  void commitOldVersion() {
    String datasetPath = tempDir.resolve("commit_old_version").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      FragmentMetadata meta = testDataset.createNewFragment(123, 20);
      FragmentOperation.Append appendOp = new FragmentOperation.Append(Arrays.asList(meta));
      assertThrows(IllegalArgumentException.class, () -> {
        Dataset.commit(allocator, datasetPath, appendOp, Optional.of(0L));
      });
    }
  }

  @Test
  void appendWithoutFragment() {
    String datasetPath = tempDir.resolve("append_without_fragment").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      assertThrows(IllegalArgumentException.class, () -> {
        new FragmentOperation.Append(new ArrayList<>());
      });
    }
  }
}
