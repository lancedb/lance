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

import static com.lancedb.lance.TestUtils.getSubstraitByteBuffer;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.Base64;
import java.util.Optional;

import com.lancedb.lance.ipc.DatasetScanner;
import org.apache.arrow.dataset.scanner.ScanOptions;
import org.apache.arrow.dataset.scanner.Scanner;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

public class DatasetTest {

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
  void testWriteStreamAndOpenPath() throws IOException, URISyntaxException {
    String datasetPath = tempDir.resolve("write_stream").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
       TestUtils.RandomAccessDataset testDataset = new TestUtils.RandomAccessDataset(allocator, datasetPath);
       testDataset.createDatasetAndValidate();
       testDataset.openDatasetAndValidate();
    }
  }

  @Test
  void testCreateEmptyDataset() {
    String datasetPath = tempDir.resolve("new_empty_dataset").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
    }
  }

  @Test
  void testOpenInvalidPath() {
    String validPath = tempDir.resolve("Invalid_dataset").toString();
    assertThrows(
        RuntimeException.class,
        () -> {
          dataset = Dataset.open(validPath, new RootAllocator());
        });
  }


  @Test
  void testDatasetVersion() {
    String datasetPath = tempDir.resolve("dataset_version").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        assertEquals(1, dataset.version());
        assertEquals(1, dataset.latestVersion());
        try (Dataset dataset2 = testDataset.write(1, 5)) {
          assertEquals(1, dataset.version());
          assertEquals(2, dataset.latestVersion());
          assertEquals(2, dataset2.version());
          assertEquals(2, dataset2.latestVersion());
          try (Dataset dataset3 = testDataset.write(2, 3)) {
            assertEquals(1, dataset.version());
            assertEquals(3, dataset.latestVersion());
            assertEquals(2, dataset2.version());
            assertEquals(3, dataset2.latestVersion());
            assertEquals(3, dataset3.version());
            assertEquals(3, dataset3.latestVersion());
          }
        }
      }
    }
  }

  @Test
  void testDatasetScannerSchema() throws IOException, URISyntaxException {
    String datasetPath = tempDir.resolve("dataset_scan").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.RandomAccessDataset testDataset = new TestUtils.RandomAccessDataset(allocator, datasetPath);
      testDataset.createDatasetAndValidate();

      try (var dataset = Dataset.open(datasetPath, allocator)) {
        var scanner = dataset.newScan(new ScanOptions(1024));
        var schema = scanner.schema();
        assertEquals(testDataset.getSchema(), schema);

        try (var datasetReader = scanner.scanBatches()) {
          var batchCount = 0;
          while (datasetReader.loadNextBatch()) {
            datasetReader.getVectorSchemaRoot();
            batchCount++;
          }
          assert (batchCount > 0);
        }
      }
    }
  }

  @Test
  @Disabled
  void testDatasetScannerSubstrait() throws IOException {
    String datasetPath = tempDir.resolve("dataset_scan_subtrait").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 40)) {
        // TODO(lu): Create a valid substrait
        ByteBuffer substraitExpressionFilter = getSubstraitByteBuffer("id < 20");
        ScanOptions options = new ScanOptions.Builder(/*batchSize*/ 1024)
            .columns(Optional.empty())
            .substraitFilter(substraitExpressionFilter)
            .build();
        Scanner scanner = dataset.newScan(options);
        try (ArrowReader reader = scanner.scanBatches()) {
          assertEquals(dataset.getSchema().getFields(), reader.getVectorSchemaRoot().getSchema().getFields());
          int rowcount = 0;
          while (reader.loadNextBatch()) {
            rowcount += reader.getVectorSchemaRoot().getRowCount();
          }
          assertEquals(40, rowcount);
        }
      }
    }
  }
}
