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
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TruncateTest extends OperationTestBase {

  @Test
  void testTruncateTable(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testTruncate").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // Append some data
      int rowCount = 20;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(rowCount);
      try (Transaction txn =
              new Transaction.Builder()
                  .readVersion(dataset.version())
                  .operation(
                      Append.builder()
                          .fragments(java.util.Collections.singletonList(fragmentMeta))
                          .build())
                  .build();
          Dataset ds1 = new CommitBuilder(dataset).execute(txn)) {
        assertEquals(rowCount, ds1.countRows());

        // Truncate to empty while preserving schema
        ds1.truncateTable();
        assertEquals(0, ds1.countRows());

        try (org.lance.ipc.LanceScanner scanner = ds1.newScan()) {
          Schema schemaRes = scanner.schema();
          assertEquals(testDataset.getSchema(), schemaRes);
        }
      }
    }
  }
}
