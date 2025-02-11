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

import com.lancedb.lance.ipc.ColumnOrdering;
import com.lancedb.lance.ipc.LanceScanner;
import com.lancedb.lance.ipc.ScanOptions;

import org.apache.arrow.dataset.scanner.Scanner;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ScannerTest {
  @TempDir static Path tempDir; // Temporary directory for the tests
  private static Dataset dataset;

  @BeforeAll
  static void setup() {}

  @AfterAll
  static void tearDown() {
    // Cleanup resources used by the tests
    if (dataset != null) {
      dataset.close();
    }
  }

  @Test
  void testDatasetScanner() throws IOException {
    String datasetPath = tempDir.resolve("dataset_scanner").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 40;
      int batchRows = 20;
      try (Dataset dataset = testDataset.write(1, totalRows)) {
        Scanner scanner = dataset.newScan(batchRows);
        testDataset.validateScanResults(dataset, scanner, totalRows, batchRows);
      }
    }
  }

  @Test
  void testDatasetScannerFilter() throws Exception {
    String datasetPath = tempDir.resolve("dataset_scanner_filter").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      // write id with value from 0 to 39
      try (Dataset dataset = testDataset.write(1, 40)) {
        try (Scanner scanner =
            dataset.newScan(new ScanOptions.Builder().filter("id < 20").build())) {
          testDataset.validateScanResults(dataset, scanner, 20, 20);
        }
      }
    }
  }

  @Test
  void testDatasetScannerColumns() throws Exception {
    String datasetPath = tempDir.resolve("dataset_scanner_columns").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 40;
      int batchRows = 20;
      try (Dataset dataset = testDataset.write(1, totalRows)) {
        try (Scanner scanner =
            dataset.newScan(
                new ScanOptions.Builder()
                    .batchSize(batchRows)
                    .columns(Arrays.asList("id"))
                    .build())) {
          try (ArrowReader reader = scanner.scanBatches()) {
            VectorSchemaRoot root = reader.getVectorSchemaRoot();
            int index = 0;
            while (reader.loadNextBatch()) {
              List<FieldVector> fieldVectors = root.getFieldVectors();
              assertEquals(1, fieldVectors.size());
              FieldVector fieldVector = fieldVectors.get(0);
              assertEquals(ArrowType.ArrowTypeID.Int, fieldVector.getField().getType().getTypeID());
              assertEquals(batchRows, fieldVector.getValueCount());
              IntVector vector = (IntVector) fieldVector;
              for (int i = 0; i < batchRows; i++) {
                assertEquals(index, vector.get(i));
                index++;
              }
            }
          }
        }
      }
    }
  }

  @Test
  void testDatasetScannerSchema() throws Exception {
    String datasetPath = tempDir.resolve("dataset_scanner_schema").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 40;
      try (Dataset dataset = testDataset.write(1, totalRows)) {
        try (Scanner scanner =
            dataset.newScan(
                new ScanOptions.Builder()
                    .batchSize(totalRows)
                    .columns(Arrays.asList("id"))
                    .build())) {
          Schema expectedSchema =
              new Schema(Arrays.asList(Field.nullable("id", new ArrowType.Int(32, true))));
          assertEquals(expectedSchema, scanner.schema());
        }
      }
    }
  }

  @Test
  void testDatasetScannerCountRows() throws Exception {
    String datasetPath = tempDir.resolve("dataset_scanner_count").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      // write id with value from 0 to 39
      try (Dataset dataset = testDataset.write(1, 40)) {
        try (LanceScanner scanner =
            dataset.newScan(
                new ScanOptions.Builder()
                    .columns(Arrays.asList())
                    .withRowId(true)
                    .filter("id < 20")
                    .build())) {
          assertEquals(20, scanner.countRows());
        }
      }
    }
  }

  @Test
  void testFragmentScanner() throws Exception {
    String datasetPath = tempDir.resolve("fragment_scanner").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 40;
      int batchRows = 20;
      try (Dataset dataset = testDataset.write(1, totalRows)) {
        Fragment fragment = dataset.getFragments().get(0);
        try (Scanner scanner = fragment.newScan(batchRows)) {
          testDataset.validateScanResults(dataset, scanner, totalRows, batchRows);
        }
      }
    }
  }

  @Test
  void testFragmentScannerFilter() throws Exception {
    String datasetPath = tempDir.resolve("fragment_scanner_filter").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      // write id with value from 0 to 39
      try (Dataset dataset = testDataset.write(1, 40)) {
        Fragment fragment = dataset.getFragments().get(0);
        try (Scanner scanner =
            fragment.newScan(new ScanOptions.Builder().filter("id < 20").build())) {
          testDataset.validateScanResults(dataset, scanner, 20, 20);
        }
      }
    }
  }

  @Test
  void testFragmentScannerColumns() throws Exception {
    String datasetPath = tempDir.resolve("fragment_scanner_columns").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 40;
      int batchRows = 20;
      try (Dataset dataset = testDataset.write(1, totalRows)) {
        Fragment fragment = dataset.getFragments().get(0);
        try (Scanner scanner =
            fragment.newScan(
                new ScanOptions.Builder()
                    .batchSize(batchRows)
                    .columns(Arrays.asList("id"))
                    .build())) {
          try (ArrowReader reader = scanner.scanBatches()) {
            VectorSchemaRoot root = reader.getVectorSchemaRoot();
            int index = 0;
            while (reader.loadNextBatch()) {
              List<FieldVector> fieldVectors = root.getFieldVectors();
              assertEquals(1, fieldVectors.size());
              FieldVector fieldVector = fieldVectors.get(0);
              assertEquals(ArrowType.ArrowTypeID.Int, fieldVector.getField().getType().getTypeID());
              assertEquals(batchRows, fieldVector.getValueCount());
              IntVector vector = (IntVector) fieldVector;
              for (int i = 0; i < batchRows; i++) {
                assertEquals(index, vector.get(i));
                index++;
              }
            }
          }
        }
      }
    }
  }

  @Test
  void testScanFragment() throws Exception {
    String datasetPath = tempDir.resolve("fragment_scanner_single_fragment").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      FragmentMetadata metadata0 = testDataset.createNewFragment(3);
      FragmentMetadata metadata1 = testDataset.createNewFragment(5);
      FragmentMetadata metadata2 = testDataset.createNewFragment(7);
      FragmentOperation.Append appendOp =
          new FragmentOperation.Append(Arrays.asList(metadata0, metadata1, metadata2));
      try (Dataset dataset = Dataset.commit(allocator, datasetPath, appendOp, Optional.of(1L))) {
        List<Fragment> frags = dataset.getFragments();
        assertEquals(3, frags.size());
        validScanResult(dataset, frags.get(0).getId(), 3);
        validScanResult(dataset, frags.get(1).getId(), 5);
        validScanResult(dataset, frags.get(2).getId(), 7);
      }
    }
  }

  @Test
  void testScanFragments() throws Exception {
    String datasetPath = tempDir.resolve("fragments_scanner").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      FragmentMetadata metadata0 = testDataset.createNewFragment(3);
      FragmentMetadata metadata1 = testDataset.createNewFragment(5);
      FragmentMetadata metadata2 = testDataset.createNewFragment(7);
      FragmentOperation.Append appendOp =
          new FragmentOperation.Append(Arrays.asList(metadata0, metadata1, metadata2));
      try (Dataset dataset = Dataset.commit(allocator, datasetPath, appendOp, Optional.of(1L))) {
        List<Fragment> frags = dataset.getFragments();
        assertEquals(3, frags.size());
        try (Scanner scanner =
            dataset.newScan(
                new ScanOptions.Builder()
                    .batchSize(1024)
                    .fragmentIds(Arrays.asList(frags.get(1).getId(), frags.get(2).getId()))
                    .build())) {
          try (ArrowReader reader = scanner.scanBatches()) {
            assertEquals(
                dataset.getSchema().getFields(),
                reader.getVectorSchemaRoot().getSchema().getFields());
            int rowcount = 0;
            reader.loadNextBatch();
            int currentRowCount = reader.getVectorSchemaRoot().getRowCount();
            assertEquals(5, currentRowCount);
            rowcount += currentRowCount;
            reader.loadNextBatch();
            currentRowCount = reader.getVectorSchemaRoot().getRowCount();
            assertEquals(7, currentRowCount);
            rowcount += currentRowCount;
            assertEquals(12, rowcount);
          }
        }
      }
    }
  }

  @Test
  void testDatasetScannerLimit() throws Exception {
    String datasetPath = tempDir.resolve("dataset_scanner_limit").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 100;
      int limit = 50;
      try (Dataset dataset = testDataset.write(1, totalRows)) {
        try (Scanner scanner = dataset.newScan(new ScanOptions.Builder().limit(limit).build())) {
          testDataset.validateScanResults(dataset, scanner, limit, limit);
        }
      }
    }
  }

  @Test
  void testDatasetScannerOffset() throws Exception {
    String datasetPath = tempDir.resolve("dataset_scanner_offset").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 100;
      int offset = 50;
      try (Dataset dataset = testDataset.write(1, totalRows)) {
        try (Scanner scanner = dataset.newScan(new ScanOptions.Builder().offset(offset).build())) {
          testDataset.validateScanResults(
              dataset, scanner, totalRows - offset, totalRows - offset, offset);
        }
      }
    }
  }

  @Test
  void testDatasetScannerWithRowId() throws Exception {
    String datasetPath = tempDir.resolve("dataset_scanner_with_row_id").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 50;
      try (Dataset dataset = testDataset.write(1, totalRows)) {
        try (Scanner scanner = dataset.newScan(new ScanOptions.Builder().withRowId(true).build())) {
          try (ArrowReader reader = scanner.scanBatches()) {
            VectorSchemaRoot root = reader.getVectorSchemaRoot();
            assertTrue(
                root.getSchema().getFields().stream()
                    .anyMatch(field -> field.getName().equals("_rowid")));
            while (reader.loadNextBatch()) {
              List<FieldVector> fieldVectors = root.getFieldVectors();
              assertTrue(
                  fieldVectors.stream().anyMatch(vector -> vector.getName().equals("_rowid")));
            }
          }
        }
      }
    }
  }

  @Test
  void testDatasetScannerBatchReadahead() throws Exception {
    String datasetPath = tempDir.resolve("dataset_scanner_batch_readahead").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 1000;
      int batchSize = 100;
      int batchReadahead = 5;
      try (Dataset dataset = testDataset.write(1, totalRows)) {
        try (LanceScanner scanner =
            dataset.newScan(
                new ScanOptions.Builder()
                    .batchSize(batchSize)
                    .batchReadahead(batchReadahead)
                    .build())) {
          // This test is more about ensuring that the batchReadahead parameter is accepted
          // and doesn't cause errors. The actual effect of batchReadahead might not be
          // directly observable in this test.
          try (ArrowReader reader = scanner.scanBatches()) {
            int rowCount = 0;
            while (reader.loadNextBatch()) {
              rowCount += reader.getVectorSchemaRoot().getRowCount();
            }
            assertEquals(totalRows, rowCount);
          }
        }
      }
    }
  }

  @Test
  void testDatasetScannerSortBy() throws Exception {
    String datasetPath = tempDir.resolve("testDatasetScannerSortBy").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.writeSortByDataset(1)) {
        ColumnOrdering.Builder nameBuilder = new ColumnOrdering.Builder();
        nameBuilder.setColumnName("name");
        nameBuilder.setAscending(true);
        nameBuilder.setNullFirst(false);

        ColumnOrdering.Builder idBuilder = new ColumnOrdering.Builder();
        idBuilder.setColumnName("id");
        idBuilder.setAscending(false);
        idBuilder.setNullFirst(true);

        List<ColumnOrdering> columnOrderings =
            Arrays.asList(nameBuilder.build(), idBuilder.build());
        ScanOptions.Builder scanOptionBuilder = new ScanOptions.Builder();
        scanOptionBuilder
            .columns(Arrays.asList("name", "id"))
            .limit(10)
            .setColumnOrderings(columnOrderings);
        ScanOptions scanOptions = scanOptionBuilder.build();
        try (Scanner scanner = dataset.newScan(scanOptions)) {
          try (ArrowReader reader = scanner.scanBatches()) {
            while (reader.loadNextBatch()) {
              List<FieldVector> fieldVectors = reader.getVectorSchemaRoot().getFieldVectors();
              VarCharVector nameVector = (VarCharVector) fieldVectors.get(0);
              /* dataset context
               * i: |  id   | name | :i
               * 1: |  1    |  P0  | :0
               * 2: | null  |  P1  | :1
               * 3: |  2    |  P2  | :2
               * 5: | null  |  P3  | :3
               * 4: |  2    |  P3  | :4
               * 7: |  4    |  P4  | :5
               * 9: |  5    |  P5  | :6
               * 8: |  4    |  P5  | :7
               * 6: |  3    | null | :8
               * 0: |  0    | null | :9
               */
              assertEquals("P0", new String(nameVector.get(0)));
              assertEquals("P1", new String(nameVector.get(1)));
              assertEquals("P2", new String(nameVector.get(2)));
              assertEquals("P3", new String(nameVector.get(3)));
              assertEquals("P3", new String(nameVector.get(4)));
              assertEquals("P4", new String(nameVector.get(5)));
              assertEquals("P5", new String(nameVector.get(6)));
              assertEquals("P5", new String(nameVector.get(7)));
              assertTrue(nameVector.isNull(8));
              assertTrue(nameVector.isNull(9));

              IntVector idVector = (IntVector) fieldVectors.get(1);
              assertEquals(1, idVector.get(0));
              assertTrue(idVector.isNull(1));
              assertEquals(2, idVector.get(2));
              assertTrue(idVector.isNull(3));
              assertEquals(2, idVector.get(4));
              assertEquals(4, idVector.get(5));
              assertEquals(5, idVector.get(6));
              assertEquals(4, idVector.get(7));
              assertEquals(3, idVector.get(8));
              assertEquals(0, idVector.get(9));
            }
          }
        }
      }
    }
  }

  @Test
  void testDatasetScannerCombinedParams() throws Exception {
    String datasetPath = tempDir.resolve("dataset_scanner_combined_params").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 600;
      int limit = 200;
      int offset = 300;
      int batchSize = 50;
      try (Dataset dataset = testDataset.write(1, totalRows)) {
        try (Scanner scanner =
            dataset.newScan(
                new ScanOptions.Builder()
                    .limit(limit)
                    .offset(offset)
                    .withRowId(true)
                    .batchSize(batchSize)
                    .batchReadahead(3)
                    .build())) {
          try (ArrowReader reader = scanner.scanBatches()) {
            VectorSchemaRoot root = reader.getVectorSchemaRoot();
            List<String> fieldNames =
                root.getSchema().getFields().stream()
                    .map(Field::getName)
                    .collect(Collectors.toList());
            assertTrue(fieldNames.contains("_rowid"), "Schema should contain _rowid column");
            assertTrue(fieldNames.contains("id"), "Schema should contain id column");

            int rowCount = 0;
            int expectedIdStart = offset;
            while (reader.loadNextBatch()) {
              List<FieldVector> fieldVectors = root.getFieldVectors();
              assertTrue(
                  fieldVectors.stream().anyMatch(vector -> vector.getName().equals("_rowid")));
              IntVector idVector = (IntVector) root.getVector("id");
              int batchRowCount = root.getRowCount();
              rowCount += batchRowCount;
              assertTrue(batchRowCount <= batchSize, "Batch size should not exceed " + batchSize);

              for (int i = 0; i < batchRowCount; i++) {
                int expectedId = expectedIdStart + i;
                assertEquals(
                    expectedId,
                    idVector.get(i),
                    "Mismatch at row "
                        + (rowCount - batchRowCount + i)
                        + ". Expected: "
                        + expectedId
                        + ", Actual: "
                        + idVector.get(i));
              }
              expectedIdStart += batchRowCount;
            }
            assertEquals(limit, rowCount, "Total rows should match the limit");
          }
        }
      }
    }
  }

  private void validScanResult(Dataset dataset, int fragmentId, int rowCount) throws Exception {
    try (Scanner scanner =
        dataset.newScan(
            new ScanOptions.Builder()
                .batchSize(1024)
                .fragmentIds(Arrays.asList(fragmentId))
                .build())) {
      try (ArrowReader reader = scanner.scanBatches()) {
        assertEquals(
            dataset.getSchema().getFields(), reader.getVectorSchemaRoot().getSchema().getFields());
        reader.loadNextBatch();
        assertEquals(rowCount, reader.getVectorSchemaRoot().getRowCount());
        assertFalse(reader.loadNextBatch());
      }
    }
  }
}
