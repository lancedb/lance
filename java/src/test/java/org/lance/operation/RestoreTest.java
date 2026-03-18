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
package org.lance.operation;

import org.lance.CommitBuilder;
import org.lance.Dataset;
import org.lance.FragmentMetadata;
import org.lance.TestUtils;
import org.lance.Transaction;

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
      try (Transaction appendTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
              .build()) {
        try (Dataset modifiedDataset = new CommitBuilder(dataset).execute(appendTxn)) {
          // Verify the dataset was modified
          long newVersion = modifiedDataset.version();
          assertEquals(initialVersion + 1, newVersion);
          assertEquals(rowCount, modifiedDataset.countRows());

          // Restore to the initial version
          try (Transaction restoreTxn =
              new Transaction.Builder()
                  .readVersion(modifiedDataset.version())
                  .operation(new Restore.Builder().version(initialVersion).build())
                  .build()) {
            try (Dataset restoredDataset = new CommitBuilder(modifiedDataset).execute(restoreTxn)) {
              // Verify the dataset was restored to the initial version, but the version increases
              assertEquals(initialVersion + 2, restoredDataset.version());
              // Initial dataset had 0 rows
              assertEquals(0, restoredDataset.countRows());
            }
          }
        }
      }
    }
  }
}
