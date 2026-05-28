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

import org.lance.ipc.LanceScanner;
import org.lance.ipc.ScanOptions;
import org.lance.update.UpdateParams;
import org.lance.update.UpdateResult;

import com.google.common.collect.ImmutableMap;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.UInt8Vector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

public class UpdateTest {
  @TempDir private Path tempDir;
  private RootAllocator allocator;
  private TestUtils.SimpleTestDataset testDataset;
  private Dataset dataset;

  @BeforeEach
  public void setup() {
    String datasetPath = tempDir.resolve(UUID.randomUUID().toString()).toString();
    allocator = new RootAllocator(Long.MAX_VALUE);
    testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
    testDataset.createEmptyDataset().close();
    dataset = testDataset.write(1, 5);
  }

  @AfterEach
  public void tearDown() {
    if (dataset != null) {
      dataset.close();
    }
    allocator.close();
  }

  @Test
  public void testUpdateAllRows() {
    UpdateResult result = dataset.update(new UpdateParams(ImmutableMap.of("name", "'updated'")));

    Assertions.assertEquals(5, result.getNumRowsUpdated());
    try (Dataset newDataset = result.getDataset()) {
      List<String> names = readNames(newDataset);
      Assertions.assertEquals(5, names.size());
      for (String name : names) {
        Assertions.assertEquals("updated", name);
      }
    }
  }

  @Test
  public void testUpdateWithWhere() {
    UpdateResult result =
        dataset.update(new UpdateParams(ImmutableMap.of("name", "'updated'")).withWhere("id = 2"));

    Assertions.assertEquals(1, result.getNumRowsUpdated());
    try (Dataset newDataset = result.getDataset()) {
      // id=2 should be 'updated'; others remain "Person <i>".
      try (LanceScanner scanner =
          newDataset.newScan(
              new ScanOptions.Builder().columns(java.util.Arrays.asList("id", "name")).build())) {
        try (ArrowReader reader = scanner.scanBatches()) {
          int seen = 0;
          while (reader.loadNextBatch()) {
            VectorSchemaRoot batch = reader.getVectorSchemaRoot();
            IntVector idVector = (IntVector) batch.getVector("id");
            VarCharVector nameVector = (VarCharVector) batch.getVector("name");
            for (int i = 0; i < batch.getRowCount(); i++) {
              int id = idVector.get(i);
              String name = new String(nameVector.get(i));
              if (id == 2) {
                Assertions.assertEquals("updated", name);
              } else {
                Assertions.assertEquals("Person " + id, name);
              }
              seen++;
            }
          }
          Assertions.assertEquals(5, seen);
        }
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    }
  }

  @Test
  public void testUpdateByRowId() throws Exception {
    // Read the stable row id of one row and update by `_rowid`.
    long targetRowId;
    int targetId;
    try (LanceScanner scanner =
        dataset.newScan(
            new ScanOptions.Builder()
                .columns(java.util.Arrays.asList("id"))
                .withRowId(true)
                .build())) {
      try (ArrowReader reader = scanner.scanBatches()) {
        Assertions.assertTrue(reader.loadNextBatch());
        VectorSchemaRoot batch = reader.getVectorSchemaRoot();
        UInt8Vector rowIdVector = (UInt8Vector) batch.getVector("_rowid");
        IntVector idVector = (IntVector) batch.getVector("id");
        targetRowId = rowIdVector.get(2);
        targetId = idVector.get(2);
      }
    }

    UpdateResult result =
        dataset.update(
            new UpdateParams(ImmutableMap.of("name", "'updated'"))
                .withWhere("_rowid = " + targetRowId));

    Assertions.assertEquals(1, result.getNumRowsUpdated());
    try (Dataset newDataset = result.getDataset()) {
      try (LanceScanner scanner =
          newDataset.newScan(
              new ScanOptions.Builder().columns(java.util.Arrays.asList("id", "name")).build())) {
        try (ArrowReader reader = scanner.scanBatches()) {
          int updated = 0;
          while (reader.loadNextBatch()) {
            VectorSchemaRoot batch = reader.getVectorSchemaRoot();
            IntVector idVector = (IntVector) batch.getVector("id");
            VarCharVector nameVector = (VarCharVector) batch.getVector("name");
            for (int i = 0; i < batch.getRowCount(); i++) {
              int id = idVector.get(i);
              String name = new String(nameVector.get(i));
              if (id == targetId) {
                Assertions.assertEquals("updated", name);
                updated++;
              } else {
                Assertions.assertEquals("Person " + id, name);
              }
            }
          }
          Assertions.assertEquals(1, updated);
        }
      }
    }
  }

  @Test
  public void testUpdateByRowIdInList() throws Exception {
    List<Long> targetRowIds = new ArrayList<>();
    List<Integer> targetIds = new ArrayList<>();
    try (LanceScanner scanner =
        dataset.newScan(
            new ScanOptions.Builder()
                .columns(java.util.Arrays.asList("id"))
                .withRowId(true)
                .build())) {
      try (ArrowReader reader = scanner.scanBatches()) {
        Assertions.assertTrue(reader.loadNextBatch());
        VectorSchemaRoot batch = reader.getVectorSchemaRoot();
        UInt8Vector rowIdVector = (UInt8Vector) batch.getVector("_rowid");
        IntVector idVector = (IntVector) batch.getVector("id");
        for (int idx : new int[] {0, 2, 4}) {
          targetRowIds.add(rowIdVector.get(idx));
          targetIds.add(idVector.get(idx));
        }
      }
    }

    StringBuilder inList = new StringBuilder();
    for (int i = 0; i < targetRowIds.size(); i++) {
      if (i > 0) {
        inList.append(", ");
      }
      inList.append(targetRowIds.get(i));
    }

    UpdateResult result =
        dataset.update(
            new UpdateParams(ImmutableMap.of("name", "'updated'"))
                .withWhere("_rowid IN (" + inList + ")"));

    Assertions.assertEquals(targetIds.size(), result.getNumRowsUpdated());
    try (Dataset newDataset = result.getDataset()) {
      try (LanceScanner scanner =
          newDataset.newScan(
              new ScanOptions.Builder().columns(java.util.Arrays.asList("id", "name")).build())) {
        try (ArrowReader reader = scanner.scanBatches()) {
          int updated = 0;
          while (reader.loadNextBatch()) {
            VectorSchemaRoot batch = reader.getVectorSchemaRoot();
            IntVector idVector = (IntVector) batch.getVector("id");
            VarCharVector nameVector = (VarCharVector) batch.getVector("name");
            for (int i = 0; i < batch.getRowCount(); i++) {
              int id = idVector.get(i);
              String name = new String(nameVector.get(i));
              if (targetIds.contains(id)) {
                Assertions.assertEquals("updated", name);
                updated++;
              } else {
                Assertions.assertEquals("Person " + id, name);
              }
            }
          }
          Assertions.assertEquals(targetIds.size(), updated);
        }
      }
    }
  }

  @Test
  public void testUpdateRejectsEmptyUpdates() {
    Assertions.assertThrows(
        IllegalArgumentException.class, () -> new UpdateParams(Collections.emptyMap()));
  }

  private List<String> readNames(Dataset dataset) {
    List<String> names = new ArrayList<>();
    try (LanceScanner scanner =
        dataset.newScan(
            new ScanOptions.Builder().columns(java.util.Arrays.asList("name")).build())) {
      try (ArrowReader reader = scanner.scanBatches()) {
        while (reader.loadNextBatch()) {
          VectorSchemaRoot batch = reader.getVectorSchemaRoot();
          VarCharVector nameVector = (VarCharVector) batch.getVector("name");
          for (int i = 0; i < batch.getRowCount(); i++) {
            names.add(new String(nameVector.get(i)));
          }
        }
      }
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    return names;
  }
}
