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

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import com.lancedb.lance.ipc.LanceScanner;
import com.lancedb.lance.ipc.ScanOptions;
import org.apache.arrow.dataset.scanner.Scanner;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

public class ScannerTest {
  @TempDir
  static Path tempDir; // Temporary directory for the tests
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
      TestUtils.SimpleTestDataset testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
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
      TestUtils.SimpleTestDataset testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      // write id with value from 0 to 39
      try (Dataset dataset = testDataset.write(1, 40)) {
        try (Scanner scanner = dataset.newScan(new ScanOptions.Builder().filter("id < 20").build())) {
          testDataset.validateScanResults(dataset, scanner, 20, 20);
        }
      }
    }
  }

  @Test
  void testDatasetScannerColumns() throws Exception {
    String datasetPath = tempDir.resolve("dataset_scanner_columns").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 40;
      int batchRows = 20;
      try (Dataset dataset = testDataset.write(1, totalRows)) {
        try (Scanner scanner = dataset.newScan(new ScanOptions.Builder()
            .batchSize(batchRows).columns(List.of("id")).build())) {
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
      TestUtils.SimpleTestDataset testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 40;
      try (Dataset dataset = testDataset.write(1, totalRows)) {
        try (Scanner scanner = dataset.newScan(new ScanOptions.Builder()
            .batchSize(totalRows).columns(List.of("id")).build())) {
          Schema expectedSchema = new Schema(Arrays.asList(
              Field.nullable("id", new ArrowType.Int(32, true))
          ));
          assertEquals(expectedSchema, scanner.schema());
        }
      }
    }
  }

  @Test
  void testDatasetScannerCountRows() throws Exception {
    String datasetPath = tempDir.resolve("dataset_scanner_count").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      // write id with value from 0 to 39
      try (Dataset dataset = testDataset.write(1, 40)) {
        try (LanceScanner scanner = dataset.newScan(new ScanOptions.Builder().filter("id < 20").build())) {
          assertEquals(20, scanner.countRows());
        }
      }
    }
  }

  @Test
  void testFragmentScanner() throws Exception {
    String datasetPath = tempDir.resolve("fragment_scanner").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 40;
      int batchRows = 20;
      try (Dataset dataset = testDataset.write(1, totalRows)) {
        var fragment = dataset.getFragments().get(0);
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
      TestUtils.SimpleTestDataset testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      // write id with value from 0 to 39
      try (Dataset dataset = testDataset.write(1, 40)) {
        var fragment = dataset.getFragments().get(0);
        try (Scanner scanner = fragment.newScan(new ScanOptions.Builder().filter("id < 20").build())) {
          testDataset.validateScanResults(dataset, scanner, 20, 20);
        }
      }
    }
  }

  @Test
  void testFragmentScannerColumns() throws Exception {
    String datasetPath = tempDir.resolve("fragment_scanner_columns").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 40;
      int batchRows = 20;
      try (Dataset dataset = testDataset.write(1, totalRows)) {
        var fragment = dataset.getFragments().get(0);
        try (Scanner scanner = fragment.newScan(new ScanOptions.Builder().batchSize(batchRows).columns(List.of("id")).build())) {
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
      TestUtils.SimpleTestDataset testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int[] fragment0 = new int[]{0, 3};
      int[] fragment1 = new int[]{1, 5};
      int[] fragment2 = new int[]{2, 7};
      FragmentMetadata metadata0 = testDataset.createNewFragment(fragment0[0], fragment0[1]);
      FragmentMetadata metadata1 = testDataset.createNewFragment(fragment1[0], fragment1[1]);
      FragmentMetadata metadata2 = testDataset.createNewFragment(fragment2[0], fragment2[1]);
      FragmentOperation.Append appendOp = new FragmentOperation.Append(List.of(metadata0, metadata1, metadata2));
      try (Dataset dataset = Dataset.commit(allocator, datasetPath, appendOp, Optional.of(1L))) {
        validScanResult(dataset, fragment0[0], fragment0[1]);
        validScanResult(dataset, fragment1[0], fragment1[1]);
        validScanResult(dataset, fragment2[0], fragment2[1]);
      }
    }
  }

  @Test
  void testScanFragments() throws Exception {
    String datasetPath = tempDir.resolve("fragments_scanner").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int[] fragment0 = new int[]{0, 3};
      int[] fragment1 = new int[]{1, 5};
      int[] fragment2 = new int[]{2, 7};
      FragmentMetadata metadata0 = testDataset.createNewFragment(fragment0[0], fragment0[1]);
      FragmentMetadata metadata1 = testDataset.createNewFragment(fragment1[0], fragment1[1]);
      FragmentMetadata metadata2 = testDataset.createNewFragment(fragment2[0], fragment2[1]);
      FragmentOperation.Append appendOp = new FragmentOperation.Append(List.of(metadata0, metadata1, metadata2));
      try (Dataset dataset = Dataset.commit(allocator, datasetPath, appendOp, Optional.of(1L))) {
        try (Scanner scanner = dataset.newScan(new ScanOptions.Builder().batchSize(1024).fragmentIds(List.of(1, 2)).build())) {
          try (ArrowReader reader = scanner.scanBatches()) {
            assertEquals(dataset.getSchema().getFields(), reader.getVectorSchemaRoot().getSchema().getFields());
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

  private void validScanResult(Dataset dataset, int fragmentId, int rowCount) throws Exception {
    try (Scanner scanner = dataset.newScan(new ScanOptions.Builder().batchSize(1024).fragmentIds(List.of(fragmentId)).build())) {
      try (ArrowReader reader = scanner.scanBatches()) {
        assertEquals(dataset.getSchema().getFields(), reader.getVectorSchemaRoot().getSchema().getFields());
        reader.loadNextBatch();
        assertEquals(rowCount, reader.getVectorSchemaRoot().getRowCount());
        assertFalse(reader.loadNextBatch());
      }
    }
  }

}
