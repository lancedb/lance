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
}
