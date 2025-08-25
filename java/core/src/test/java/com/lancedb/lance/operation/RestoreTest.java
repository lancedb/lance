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
package com.lancedb.lance.operation;

import com.lancedb.lance.Dataset;
import com.lancedb.lance.FragmentMetadata;
import com.lancedb.lance.TestUtils;
import com.lancedb.lance.Transaction;

import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class RestoreTest extends OperationTestBase {

  @Test
  void testRestore(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testRestore").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // Record the initial version
      long initialVersion = dataset.version();

      // Append data to create a new version
      int rowCount = 20;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(rowCount);
      Transaction transaction =
          dataset
              .newTransactionBuilder()
              .operation(
                  Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
              .build();
      try (Dataset modifiedDataset = transaction.commit()) {
        // Verify the dataset was modified
        long newVersion = modifiedDataset.version();
        assertEquals(initialVersion + 1, newVersion);
        assertEquals(rowCount, modifiedDataset.countRows());

        // Restore to the initial version
        Transaction restoreTransaction =
            modifiedDataset
                .newTransactionBuilder()
                .operation(new Restore.Builder().version(initialVersion).build())
                .build();
        try (Dataset restoredDataset = restoreTransaction.commit()) {
          // Verify the dataset was restored to the initial version, but the version increases
          assertEquals(initialVersion + 2, restoredDataset.version());
          // Initial dataset had 0 rows
          assertEquals(0, restoredDataset.countRows());
          assertEquals(restoreTransaction, restoredDataset.readTransaction().orElse(null));
        }
      }
    }
  }
}
