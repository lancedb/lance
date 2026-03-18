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

import org.lance.fragment.FragmentMergeResult;
import org.lance.ipc.LanceScanner;
import org.lance.ipc.ScanOptions;
import org.lance.operation.Merge;
import org.lance.operation.Update;

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.UInt8Vector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class FragmentTest {
  @Test
  void testFragmentCreateFfiArray(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("new_fragment_array").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      testDataset.createNewFragment(20);
    }
  }

  @Test
  void testFragmentCreate(@TempDir Path tempDir) throws Exception {
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
  void commitWithoutVersion(@TempDir Path tempDir) {
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
  void appendWithoutFragment(@TempDir Path tempDir) {
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
  void testOverwriteCommit(@TempDir Path tempDir) throws Exception {
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
  void testEmptyFragments(@TempDir Path tempDir) {
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
  void testMultiFragments(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testMultiFragments").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      List<FragmentMetadata> fragments = testDataset.createNewFragment(20, 10);
      assertEquals(2, fragments.size());
    }
  }

  @Test
  void testDeleteRows(@TempDir Path tempDir) throws IOException {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset();

      int totalRows = 100;
      try (Dataset dataset2 = testDataset.write(1, totalRows)) {
        assertEquals(totalRows, dataset2.countRows());

        Fragment fragment = dataset2.getFragments().get(0);
        List<Integer> rowIndexes = readAllRows(fragment);

        // Case 1. Test delete some rows

        Collections.shuffle(rowIndexes);
        int deleteCount = rowIndexes.size() / 2;
        FragmentMetadata updateFragment = fragment.deleteRows(rowIndexes.subList(0, deleteCount));

        assertNotNull(updateFragment);
        assertNotNull(updateFragment.getDeletionFile());

        Update update =
            Update.builder().updatedFragments(Collections.singletonList(updateFragment)).build();
        Dataset dataset3;
        try (Transaction txn =
            new Transaction.Builder().readVersion(dataset2.version()).operation(update).build()) {
          dataset3 = new CommitBuilder(dataset2).execute(txn);
        }

        assertEquals(totalRows - deleteCount, dataset3.countRows());

        // Case 2. Test more some rows
        fragment = dataset3.getFragments().get(0);
        rowIndexes = readAllRows(fragment);

        int deleteCount2 = rowIndexes.size() / 2;
        updateFragment = fragment.deleteRows(rowIndexes.subList(0, deleteCount2));

        assertNotNull(updateFragment);
        assertNotNull(updateFragment.getDeletionFile());

        update =
            Update.builder().updatedFragments(Collections.singletonList(updateFragment)).build();
        Dataset dataset4;
        try (Transaction txn =
            new Transaction.Builder().readVersion(dataset3.version()).operation(update).build()) {
          dataset4 = new CommitBuilder(dataset3).execute(txn);
        }
        assertEquals(totalRows - deleteCount - deleteCount2, dataset4.countRows());

        // Case 3. Test delete all rows

        fragment = dataset4.getFragments().get(0);
        rowIndexes = readAllRows(fragment);

        updateFragment = fragment.deleteRows(rowIndexes);

        assertNull(updateFragment);

        update =
            Update.builder()
                .removedFragmentIds(Collections.singletonList(Long.valueOf(fragment.getId())))
                .build();
        Dataset dataset5;
        try (Transaction txn =
            new Transaction.Builder().readVersion(dataset4.version()).operation(update).build()) {
          dataset5 = new CommitBuilder(dataset4).execute(txn);
        }

        assertEquals(0, dataset5.countRows());
      }
    }
  }

  private List<Integer> readAllRows(Fragment fragment) throws IOException {
    List<Long> rowAddrs = new ArrayList<>();

    LanceScanner scanner = fragment.newScan(new ScanOptions.Builder().withRowAddress(true).build());
    try (ArrowReader reader = scanner.scanBatches()) {
      while (reader.loadNextBatch()) {
        VectorSchemaRoot root = reader.getVectorSchemaRoot();
        UInt8Vector rowAddressVector = (UInt8Vector) root.getVector("_rowaddr");
        for (int i = 0; i < rowAddressVector.getValueCount(); i++) {
          rowAddrs.add(rowAddressVector.get(i));
        }
      }
    }

    return rowAddrs.stream().map(RowAddress::rowIndex).collect(Collectors.toList());
  }

  @Test
  void testMergeColumns(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testMergeColumns").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.MergeColumnTestDataset testDataset =
          new TestUtils.MergeColumnTestDataset(allocator, datasetPath);
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

        FragmentMergeResult mergeResult = testDataset.mergeColumn(fragment, 10);

        try (Transaction transaction =
            new Transaction.Builder()
                .readVersion(dataset.version())
                .operation(
                    Merge.builder()
                        .fragments(Collections.singletonList(mergeResult.getFragmentMetadata()))
                        .schema(mergeResult.getSchema().asArrowSchema())
                        .build())
                .build()) {
          try (Dataset newDs = new CommitBuilder(dataset).execute(transaction)) {
            assertEquals(3, newDs.version());
            assertEquals(3, newDs.latestVersion());
            Fragment newFrag = newDs.getFragments().get(0);
            try (LanceScanner scanner = newFrag.newScan()) {
              Schema schemaRes = scanner.schema();
              assertTrue(
                  schemaRes.getFields().stream()
                      .anyMatch(field -> field.getName().equals("new_col1")));
              assertTrue(
                  schemaRes.getFields().stream()
                      .anyMatch(field -> field.getName().equals("new_col2")));

              try (ArrowReader reader = scanner.scanBatches()) {
                assertTrue(reader.loadNextBatch());
                VectorSchemaRoot root = reader.getVectorSchemaRoot();
                VarCharVector newCol1Vec = (VarCharVector) root.getVector("new_col1");
                VarCharVector newCol2Vec = (VarCharVector) root.getVector("new_col2");
                assertEquals(21, newCol2Vec.getValueCount());

                // The first 10 rows are not null
                assertNotNull(newCol1Vec.get(9));
                // Remaining rows are null
                assertNull(newCol1Vec.get(10));
              }
            }
          }
        }
      }
    }
  }
}
