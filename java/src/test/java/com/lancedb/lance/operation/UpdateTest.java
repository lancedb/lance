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
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

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
      dataset = testDataset.createEmptyDataset();
      /* dataset content
       * _rowid |   id   |     name     | timeStamp |
       *   0:   |    0   |  "Person 0"  |     0     |
       *   1:   |    1   |  "Person 1"  |    null   |
       *   2:   |  null  |     null     |     2     |
       *   3:   |  null  |     null     |    null   |
       *   4:   |    4   |  "Person 4"  |     4     |
       *   5:   |  null  |     null     |    null   |
       */
      int rowCount = 6;
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
      int updateRowCount = 4;
      /* source fragment content
       * _rowid |   id   |     name     |
       *   0:   |   100  |  "Update 0"  |
       *   1:   |  null  |     null     |
       *   2:   |    2   |  "Update 2"  |
       *   3:   |  null  |     null     |
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
        try (LanceScanner scanner = fragment.newScan(rowCount)) {
          List<Integer> actualIds = new ArrayList<>(rowCount);
          List<String> actualNames = new ArrayList<>(rowCount);
          List<Long> actualTimeStamps = new ArrayList<>(rowCount);
          try (ArrowReader reader = scanner.scanBatches()) {
            while (reader.loadNextBatch()) {
              VectorSchemaRoot root = reader.getVectorSchemaRoot();
              IntVector idVector = (IntVector) root.getVector("id");
              for (int i = 0; i < idVector.getValueCount(); i++) {
                actualIds.add(idVector.isNull(i) ? null : idVector.getObject(i));
              }
              VarCharVector nameVector = (VarCharVector) root.getVector("name");
              for (int i = 0; i < nameVector.getValueCount(); i++) {
                actualNames.add(nameVector.isNull(i) ? null : nameVector.getObject(i).toString());
              }
              TimeStampSecTZVector timeStampVector =
                  (TimeStampSecTZVector) root.getVector("timeStamp");
              for (int i = 0; i < timeStampVector.getValueCount(); i++) {
                actualTimeStamps.add(
                    timeStampVector.isNull(i) ? null : timeStampVector.getObject(i));
              }
            }
          }
          /* result dataset content
           * _rowid |   id   |     name     | timeStamp |
           *   0:   |   100  |  "Update 0"  |     0     |
           *   1:   |  null  |     null     |    null   |
           *   2:   |    2   |  "Update 2"  |     2     |
           *   3:   |  null  |     null     |    null   |
           *   4:   |    4   |  "Person 4"  |     4     |
           *   5:   |  null  |     null     |    null   |
           */
          List<Integer> expectIds = Arrays.asList(100, null, 2, null, 4, null);
          List<String> expectNames =
              Arrays.asList("Update 0", null, "Update 2", null, "Person 4", null);
          List<Long> expectTimeStamps = Arrays.asList(0L, null, 2L, null, 4L, null);
          assertEquals(expectIds, actualIds);
          assertEquals(expectNames, actualNames);
          assertEquals(expectTimeStamps, actualTimeStamps);
        }
      }
    }
  }
}
