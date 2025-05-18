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
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.commons.lang3.tuple.Pair;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.StandardCharsets;
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

  @Test
  void testFragmentMerge() throws Exception {
    String leftDatasetPath = tempDir.resolve("fragment_merge_left").toString();
    String rightDatasetPath = tempDir.resolve("fragment_merge_right").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, leftDatasetPath);
      testDataset.createEmptyDataset().close();

      int rowCount = 20, maxRowsPerFile = 10;
      List<FragmentMetadata> fragmentMetas =
          testDataset.createNewFragment(rowCount, maxRowsPerFile);
      FragmentOperation.Append appendOp = new FragmentOperation.Append(fragmentMetas);
      Dataset leftDataset = Dataset.commit(allocator, leftDatasetPath, appendOp, Optional.of(1L));

      Schema rightSchema =
          new Schema(
              Arrays.asList(
                  Field.nullable("id", new ArrowType.Int(32, true)),
                  Field.nullable("tag", new ArrowType.Utf8())));

      List<FragmentMetadata> rightFragmentMetas;
      try (VectorSchemaRoot root = VectorSchemaRoot.create(rightSchema, allocator)) {
        root.allocateNew();
        IntVector idVector = (IntVector) root.getVector("id");
        VarCharVector extraValueVector = (VarCharVector) root.getVector("tag");

        for (int i = 0; i < rowCount; i++) {
          idVector.setSafe(i, i);
          byte[] valueBytes = ("tag-" + i).getBytes(StandardCharsets.UTF_8);
          extraValueVector.set(i, valueBytes);
        }
        root.setRowCount(rowCount);
        rightFragmentMetas =
            Fragment.create(
                rightDatasetPath,
                allocator,
                root,
                new WriteParams.Builder().withMaxRowsPerFile(maxRowsPerFile).build());
      }
      FragmentOperation.Overwrite overwriteOp =
          new FragmentOperation.Overwrite(rightFragmentMetas, rightSchema);
      Dataset rightDataset =
          Dataset.commit(allocator, rightDatasetPath, overwriteOp, Optional.of(1L));

      assertEquals(leftDataset.getFragments().size(), rightDataset.getFragments().size());

      List<FragmentMetadata> mergedFragments = new ArrayList<>();
      List<Schema> mergedSchema = new ArrayList<>();
      for (int i = 0; i < leftDataset.getFragments().size(); ++i) {
        Fragment leftFragment = leftDataset.getFragments().get(i);
        Fragment rightFragment = rightDataset.getFragments().get(i);
        try (LanceScanner scanner = rightFragment.newScan()) {
          ArrowReader reader = scanner.scanBatches();
          Pair<FragmentMetadata, Schema> mergedInfo =
              leftFragment.merge(allocator, reader, "id", "id");
          mergedFragments.add(mergedInfo.getLeft());
          mergedSchema.add(mergedInfo.getRight());
        }
      }
      assertEquals(2, mergedFragments.size());
      assertEquals(2, mergedSchema.size());

      FragmentOperation.Merge mergedOp =
          new FragmentOperation.Merge(mergedFragments, mergedSchema.get(0));
      Dataset mergedDataset = Dataset.commit(allocator, leftDatasetPath, mergedOp, Optional.of(2L));
      assertEquals(20, mergedDataset.countRows());
      assertEquals(mergedDataset.getSchema().toJson(), mergedSchema.get(0).toJson());
    }
  }
}
