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
import org.lance.Fragment;
import org.lance.FragmentMetadata;
import org.lance.TestUtils;
import org.lance.Transaction;
import org.lance.fragment.FragmentUpdateResult;
import org.lance.ipc.LanceScanner;
import org.lance.operation.Update.UpdateMode;

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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

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
      try (Transaction appendTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
              .build()) {
        try (Dataset dataset = new CommitBuilder(this.dataset).execute(appendTxn)) {
          assertEquals(2, dataset.version());
          assertEquals(2, dataset.latestVersion());
          assertEquals(rowCount, dataset.countRows());
          assertThrows(
              IllegalArgumentException.class,
              () -> {
                try (Transaction txn =
                    new Transaction.Builder()
                        .readVersion(dataset.version())
                        .operation(Append.builder().fragments(new ArrayList<>()).build())
                        .build()) {
                  new CommitBuilder(dataset).execute(txn).close();
                }
              });
        }
      }

      dataset = Dataset.open(datasetPath, allocator);
      // Update fragments
      rowCount = 40;
      FragmentMetadata newFragment = testDataset.createNewFragment(rowCount);
      try (Transaction updateTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Update.builder()
                      .removedFragmentIds(
                          Collections.singletonList(
                              Long.valueOf(dataset.getFragments().get(0).getId())))
                      .newFragments(Collections.singletonList(newFragment))
                      .updateMode(Optional.of(UpdateMode.RewriteRows))
                      .build())
              .build()) {
        try (Dataset dataset = new CommitBuilder(this.dataset).execute(updateTxn)) {
          assertEquals(3, dataset.version());
          assertEquals(3, dataset.latestVersion());
          assertEquals(rowCount, dataset.countRows());
        }
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
      try (Transaction appendTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
              .build()) {
        try (Dataset dataset = new CommitBuilder(this.dataset).execute(appendTxn)) {
          assertEquals(2, dataset.version());
          assertEquals(2, dataset.latestVersion());
          assertEquals(rowCount, dataset.countRows());
        }
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
      try (Transaction updateTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Update.builder()
                      .updatedFragments(
                          Collections.singletonList(updateResult.getUpdatedFragment()))
                      .fieldsModified(updateResult.getFieldsModified())
                      .build())
              .build()) {
        try (Dataset dataset = new CommitBuilder(this.dataset).execute(updateTxn)) {
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


  @Test
  void testUpdateColumnsReturnsMatchedOffsets(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testUpdateColumnsMatchedOffsets").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.UpdateColumnTestDataset testDataset =
          new TestUtils.UpdateColumnTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      int rowCount = 6;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(rowCount);
      try (Transaction appendTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
              .build()) {
        try (Dataset ds = new CommitBuilder(this.dataset).execute(appendTxn)) {
          assertEquals(rowCount, ds.countRows());
        }
      }

      dataset = Dataset.open(datasetPath, allocator);
      Fragment targetFragment = dataset.getFragments().get(0);
      int updateRowCount = 4;
      FragmentUpdateResult updateResult = testDataset.updateColumn(targetFragment, updateRowCount);

      // Verify matchedOffsets is populated
      long[] matchedOffsets = updateResult.getMatchedOffsets();
      assertNotNull(matchedOffsets, "matchedOffsets should not be null");
      // The update sends _rowid [0,1,2,3] for a 6-row fragment.
      // Rows 0,1,2,3 all exist, so all 4 should be matched.
      assertTrue(matchedOffsets.length > 0, "matchedOffsets should not be empty");
      assertEquals(updateRowCount, matchedOffsets.length,
          "All update rows should match");
    }
  }

  @Test
  void testUpdateWithUpdatedRowOffsets(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testUpdateWithOffsets").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.UpdateColumnTestDataset testDataset =
          new TestUtils.UpdateColumnTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      int rowCount = 6;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(rowCount);
      try (Transaction appendTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
              .build()) {
        try (Dataset ds = new CommitBuilder(this.dataset).execute(appendTxn)) {
          assertEquals(rowCount, ds.countRows());
        }
      }

      dataset = Dataset.open(datasetPath, allocator);
      Fragment targetFragment = dataset.getFragments().get(0);
      int updateRowCount = 4;
      FragmentUpdateResult updateResult = testDataset.updateColumn(targetFragment, updateRowCount);

      // Build updatedRowOffsets map from matchedOffsets
      long fragmentId = dataset.getFragments().get(0).getId();
      long[] matchedOffsets = updateResult.getMatchedOffsets();
      Map<Long, long[]> offsetsMap = new HashMap<>();
      offsetsMap.put(fragmentId, matchedOffsets);

      // Commit with updatedRowOffsets via RewriteColumns mode
      try (Transaction updateTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Update.builder()
                      .updatedFragments(
                          Collections.singletonList(updateResult.getUpdatedFragment()))
                      .fieldsModified(updateResult.getFieldsModified())
                      .updateMode(Optional.of(UpdateMode.RewriteColumns))
                      .updatedRowOffsets(Optional.of(offsetsMap))
                      .build())
              .build()) {
        try (Dataset ds = new CommitBuilder(this.dataset).execute(updateTxn)) {
          assertEquals(3, ds.version());
          assertEquals(rowCount, ds.countRows());
        }
      }
    }
  }

  @Test
  void testUpdateBuilderDefaultsEmptyOffsets() {
    // Verify builder default is Optional.empty()
    Update update = Update.builder()
        .removedFragmentIds(Collections.emptyList())
        .build();
    assertEquals(Optional.empty(), update.updatedRowOffsets());
  }

  @Test
  void testUpdateBuilderWithOffsets() {
    Map<Long, long[]> offsets = new HashMap<>();
    offsets.put(1L, new long[]{0, 2, 4});
    offsets.put(2L, new long[]{1, 3, 5});

    Update update = Update.builder()
        .removedFragmentIds(Collections.emptyList())
        .updateMode(Optional.of(UpdateMode.RewriteColumns))
        .updatedRowOffsets(Optional.of(offsets))
        .build();

    assertTrue(update.updatedRowOffsets().isPresent());
    Map<Long, long[]> got = update.updatedRowOffsets().get();
    assertEquals(2, got.size());
    assertArrayEquals(new long[]{0, 2, 4}, got.get(1L));
    assertArrayEquals(new long[]{1, 3, 5}, got.get(2L));
  }

  /**
   * Backward compatibility: old Java code builds Update without updatedRowOffsets.
   * The builder default is Optional.empty(), and commit should succeed without offsets.
   * This is the RewriteRows path where offsets are not needed.
   */
  @Test
  void testBackwardCompatUpdateWithoutOffsets_RewriteRows(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testBackwardCompatRewriteRows").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      int rowCount = 20;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(rowCount);
      try (Transaction appendTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
              .build()) {
        dataset = new CommitBuilder(this.dataset).execute(appendTxn);
      }

      // Old-style Update: no updatedRowOffsets parameter at all
      int newRowCount = 30;
      FragmentMetadata newFragment = testDataset.createNewFragment(newRowCount);
      try (Transaction updateTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Update.builder()
                      .removedFragmentIds(
                          Collections.singletonList(
                              Long.valueOf(dataset.getFragments().get(0).getId())))
                      .newFragments(Collections.singletonList(newFragment))
                      .updateMode(Optional.of(UpdateMode.RewriteRows))
                      // no .updatedRowOffsets() call — old code never set this
                      .build())
              .build()) {
        try (Dataset ds = new CommitBuilder(this.dataset).execute(updateTxn)) {
          assertEquals(3, ds.version());
          assertEquals(newRowCount, ds.countRows());
        }
      }
    }
  }

  /**
   * Backward compatibility: old Java code builds Update with RewriteColumns mode
   * but without updatedRowOffsets. This was the only available path before the
   * offsets feature was added. Commit should still succeed.
   */
  @Test
  void testBackwardCompatUpdateWithoutOffsets_RewriteColumns(@TempDir Path tempDir)
      throws Exception {
    String datasetPath = tempDir.resolve("testBackwardCompatRewriteColumns").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.UpdateColumnTestDataset testDataset =
          new TestUtils.UpdateColumnTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      int rowCount = 6;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(rowCount);
      try (Transaction appendTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
              .build()) {
        dataset = new CommitBuilder(this.dataset).execute(appendTxn);
      }

      dataset = Dataset.open(datasetPath, allocator);
      Fragment targetFragment = dataset.getFragments().get(0);
      int updateRowCount = 4;
      FragmentUpdateResult updateResult = testDataset.updateColumn(targetFragment, updateRowCount);

      // Old-style commit: RewriteColumns but without updatedRowOffsets
      try (Transaction updateTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Update.builder()
                      .updatedFragments(
                          Collections.singletonList(updateResult.getUpdatedFragment()))
                      .fieldsModified(updateResult.getFieldsModified())
                      .updateMode(Optional.of(UpdateMode.RewriteColumns))
                      // no .updatedRowOffsets() — old code path
                      .build())
              .build()) {
        try (Dataset ds = new CommitBuilder(this.dataset).execute(updateTxn)) {
          assertEquals(3, ds.version());
          assertEquals(rowCount, ds.countRows());
        }
      }
    }
  }

  /**
   * Backward compatibility: old Java code builds Update without updateMode at all.
   * This was the default before UpdateMode was introduced.
   */
  @Test
  void testBackwardCompatUpdateWithoutUpdateMode(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testBackwardCompatNoMode").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      int rowCount = 20;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(rowCount);
      try (Transaction appendTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
              .build()) {
        dataset = new CommitBuilder(this.dataset).execute(appendTxn);
      }

      int newRowCount = 30;
      FragmentMetadata newFragment = testDataset.createNewFragment(newRowCount);
      // Old-style: no updateMode, no updatedRowOffsets
      try (Transaction updateTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Update.builder()
                      .removedFragmentIds(
                          Collections.singletonList(
                              Long.valueOf(dataset.getFragments().get(0).getId())))
                      .newFragments(Collections.singletonList(newFragment))
                      // no .updateMode() — defaults to Optional.empty()
                      // no .updatedRowOffsets() — defaults to Optional.empty()
                      .build())
              .build()) {
        assertEquals(Optional.empty(), ((Update) updateTxn.getOperation()).updateMode());
        assertEquals(Optional.empty(), ((Update) updateTxn.getOperation()).updatedRowOffsets());
        try (Dataset ds = new CommitBuilder(this.dataset).execute(updateTxn)) {
          assertEquals(3, ds.version());
          assertEquals(newRowCount, ds.countRows());
        }
      }
    }
  }

  /**
   * Backward compatibility: explicitly passing Optional.empty() for updatedRowOffsets
   * should behave identically to not setting it at all.
   */
  @Test
  void testBackwardCompatExplicitEmptyOffsets(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testExplicitEmptyOffsets").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.UpdateColumnTestDataset testDataset =
          new TestUtils.UpdateColumnTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      int rowCount = 6;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(rowCount);
      try (Transaction appendTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
              .build()) {
        dataset = new CommitBuilder(this.dataset).execute(appendTxn);
      }

      dataset = Dataset.open(datasetPath, allocator);
      Fragment targetFragment = dataset.getFragments().get(0);
      int updateRowCount = 4;
      FragmentUpdateResult updateResult = testDataset.updateColumn(targetFragment, updateRowCount);

      // Explicitly pass Optional.empty() — same as not calling the method
      try (Transaction updateTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Update.builder()
                      .updatedFragments(
                          Collections.singletonList(updateResult.getUpdatedFragment()))
                      .fieldsModified(updateResult.getFieldsModified())
                      .updateMode(Optional.of(UpdateMode.RewriteColumns))
                      .updatedRowOffsets(Optional.empty())
                      .build())
              .build()) {
        assertEquals(Optional.empty(), ((Update) updateTxn.getOperation()).updatedRowOffsets());
        try (Dataset ds = new CommitBuilder(this.dataset).execute(updateTxn)) {
          assertEquals(3, ds.version());
          assertEquals(rowCount, ds.countRows());
        }
      }
    }
  }
}
