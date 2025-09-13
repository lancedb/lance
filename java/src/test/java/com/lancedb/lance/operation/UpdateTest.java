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
import com.lancedb.lance.Fragment;
import com.lancedb.lance.FragmentMetadata;
import com.lancedb.lance.TestUtils;
import com.lancedb.lance.Transaction;
import com.lancedb.lance.WriteParams;
import com.lancedb.lance.fragment.FragmentUpdateResult;
import com.lancedb.lance.ipc.LanceScanner;

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.TimeStampSecTZVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class UpdateTest extends OperationTestBase {

  @Test
  void testUpdate(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testUpdate").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // Commit fragment
      int rowCount = 20;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(rowCount);
      Transaction transaction =
          dataset
              .newTransactionBuilder()
              .operation(
                  Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
              .build();

      try (Dataset dataset = transaction.commit()) {
        assertEquals(2, dataset.version());
        assertEquals(2, dataset.latestVersion());
        assertEquals(rowCount, dataset.countRows());
        assertThrows(
            IllegalArgumentException.class,
            () ->
                dataset
                    .newTransactionBuilder()
                    .operation(Append.builder().fragments(new ArrayList<>()).build())
                    .build()
                    .commit()
                    .close());
      }

      dataset = Dataset.open(datasetPath, allocator);
      // Update fragments
      rowCount = 40;
      FragmentMetadata newFragment = testDataset.createNewFragment(rowCount);
      transaction =
          dataset
              .newTransactionBuilder()
              .operation(
                  Update.builder()
                      .removedFragmentIds(
                          Collections.singletonList(
                              Long.valueOf(dataset.getFragments().get(0).getId())))
                      .newFragments(Collections.singletonList(newFragment))
                      .build())
              .build();

      try (Dataset dataset = transaction.commit()) {
        assertEquals(3, dataset.version());
        assertEquals(3, dataset.latestVersion());
        assertEquals(rowCount, dataset.countRows());

        Transaction txn = dataset.readTransaction().orElse(null);
        assertEquals(transaction, txn);
      }
    }
  }

  @Test
  void testUpdateColumns(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testUpdateColumns").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.UpdateColumnTestDataset testDataset =
          new TestUtils.UpdateColumnTestDataset(allocator, datasetPath);
      WriteParams writeParams = new WriteParams.Builder().withEnableStableRowIds(true).build();
      dataset = testDataset.createDatasetWithWriteParams(writeParams);
      int rowCount = 20;
      /* dataset content
       * _rowid |   id   |     name     | timeStamp |
       * ------------------------------------------------
       *   0:   |    0   |  "Person 0"  |     0     |
       *   1:   |    1   |  "Person 1"  |     1     |
       *   2:   |    2   |  "Person 2"  |     2     |
       *   3:   |    3   |  "Person 3"  |     3     |
       *   4:   |    4   |  "Person 4"  |     4     |
       *   5:   |    5   |  "Person 5"  |     5     |
       *   6:   |    6   |  "Person 6"  |     6     |
       *   7:   |    7   |  "Person 7"  |     7     |
       *   8:   |    8   |  "Person 8"  |     8     |
       *   9:   |    9   |  "Person 9"  |     9     |
       *  10:   |   10   |      null    |    null   |
       *  11:   |   11   |      null    |    null   |
       *  12:   |   12   |      null    |    null   |
       *  13:   |   13   |      null    |    null   |
       *  14:   |   14   |      null    |    null   |
       *  15:   |   15   |      null    |    null   |
       *  16:   |   16   |      null    |    null   |
       *  17:   |   17   |      null    |    null   |
       *  18:   |   18   |      null    |    null   |
       *  19:   |   19   |      null    |    null   |
       */
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(rowCount);
      Transaction appendTransaction =
          dataset
              .newTransactionBuilder()
              .operation(
                  Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
              .build();
      try (Dataset dataset = appendTransaction.commit()) {
        assertEquals(2, dataset.version());
        assertEquals(2, dataset.latestVersion());
        assertEquals(rowCount, dataset.countRows());
      }

      dataset = Dataset.open(datasetPath, allocator);
      Fragment targetFragment = dataset.getFragments().get(0);
      int updateRowCount = 10;
      /* sourceFragment content
       * _rowid |   id   |     name     |
       * --------------------------------
       *   0:   |    0   |  "Update 0"  |
       *   1:   |    2   |  "Update 1"  |
       *   2:   |    4   |  "Update 2"  |
       *   3:   |    6   |  "Update 3"  |
       *   4:   |    8   |  "Update 4"  |
       *   5:   |   10   |  "Update 5"  |
       *   6:   |   12   |  "Update 6"  |
       *   7:   |   14   |  "Update 7"  |
       *   8:   |   16   |  "Update 8"  |
       *   9:   |   18   |  "Update 9"  |
       */
      FragmentUpdateResult updateResult = testDataset.updateColumn(targetFragment, updateRowCount);
      Transaction updateTransaction =
          dataset
              .newTransactionBuilder()
              .operation(
                  Update.builder()
                      .updatedFragments(
                          Collections.singletonList(updateResult.getFragmentMetadata()))
                      .updatedFieldIds(updateResult.getUpdatedFieldIds())
                      .build())
              .build();
      try (Dataset dataset = updateTransaction.commit()) {
        assertEquals(3, dataset.version());
        assertEquals(3, dataset.latestVersion());
        Fragment fragment = dataset.getFragments().get(0);
        try (LanceScanner scanner = fragment.newScan(20)) {
          List<Integer> actualIds = new ArrayList<>(20);
          List<String> actualNames = new ArrayList<>(20);
          List<Long> actualTimeStamps = new ArrayList<>(20);
          try (ArrowReader reader = scanner.scanBatches()) {
            while (reader.loadNextBatch()) {
              VectorSchemaRoot root = reader.getVectorSchemaRoot();
              IntVector intVector = (IntVector) root.getVector("id");
              for (int i = 0; i < intVector.getValueCount(); i++) {
                actualIds.add(intVector.isNull(i) ? null : intVector.getObject(i));
              }
              VarCharVector stringVector = (VarCharVector) root.getVector("name");
              for (int i = 0; i < stringVector.getValueCount(); i++) {
                actualNames.add(
                    stringVector.isNull(i) ? null : stringVector.getObject(i).toString());
              }
              TimeStampSecTZVector timeStampVector =
                  (TimeStampSecTZVector) root.getVector("timeStamp");
              for (int i = 0; i < timeStampVector.getValueCount(); i++) {
                actualTimeStamps.add(stringVector.isNull(i) ? null : timeStampVector.getObject(i));
              }
            }
          }
          List<Integer> expectIds =
              IntStream.range(0, 20)
                  .mapToObj(i -> (i < 10 ? 2 * i : i))
                  .collect(Collectors.toList());
          List<String> expectNames =
              IntStream.range(0, 20)
                  .mapToObj(i -> (i < 10 ? "Update " + i : null))
                  .collect(Collectors.toList());
          List<Long> expectTimeStamps =
              LongStream.range(0, 20)
                  .mapToObj(i -> (i < 10 ? i : null))
                  .collect(Collectors.toList());
          assertEquals(expectIds, actualIds);
          assertEquals(expectNames, actualNames);
          assertEquals(expectTimeStamps, actualTimeStamps);
        }
      }
    }
  }
}
