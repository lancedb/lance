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
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class DeleteTest extends OperationTestBase {

  @Test
  void testDelete(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testDelete").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();
      // Commit fragment
      int rowCount = 20;
      FragmentMetadata fragmentMeta0 = testDataset.createNewFragment(rowCount);
      FragmentMetadata fragmentMeta1 = testDataset.createNewFragment(rowCount);
      Transaction transaction =
          dataset
              .newTransactionBuilder()
              .operation(
                  Append.builder().fragments(Arrays.asList(fragmentMeta0, fragmentMeta1)).build())
              .build();
      try (Dataset dataset = transaction.commit()) {
        assertEquals(2, dataset.version());
        assertEquals(2, dataset.latestVersion());
      }

      dataset = Dataset.open(datasetPath, allocator);

      List<Long> deletedFragmentIds =
          dataset.getFragments().stream()
              .map(t -> Long.valueOf(t.getId()))
              .collect(Collectors.toList());

      Transaction delete =
          dataset
              .newTransactionBuilder()
              .operation(
                  Delete.builder().deletedFragmentIds(deletedFragmentIds).predicate("1=1").build())
              .build();
      try (Dataset dataset = delete.commit()) {
        Transaction txn = dataset.readTransaction().get();
        Delete execDelete = (Delete) txn.operation();
        assertEquals(delete.operation(), execDelete);
        assertEquals(0, dataset.countRows());
      }
    }
  }
}
