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

import org.lance.ipc.AsyncScanner;
import org.lance.ipc.ScanOptions;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Example tests demonstrating AsyncScanner usage with CompletableFuture-based API.
 *
 * <p>AsyncScanner provides non-blocking scan operations that prevent thread starvation in Java
 * query engines like Presto/Trino.
 */
public class AsyncScannerTest {
  private static Dataset dataset;

  @BeforeAll
  static void setup() {}

  @AfterAll
  static void tearDown() {
    if (dataset != null) {
      dataset.close();
    }
  }

  /**
   * Example 1: Basic async scan with CompletableFuture.
   *
   * <p>This shows the simplest usage - create an async scanner and wait for results.
   */
  @Test
  void testBasicAsyncScan(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("async_scanner_basic").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 40;

      try (Dataset dataset = testDataset.write(1, totalRows)) {
        // Create AsyncScanner with same options as LanceScanner
        ScanOptions options = new ScanOptions.Builder().batchSize(20L).build();

        try (AsyncScanner scanner = AsyncScanner.create(dataset, options, allocator)) {
          // Start async scan - returns CompletableFuture<ArrowReader>
          CompletableFuture<ArrowReader> future = scanner.scanBatchesAsync();

          // Wait for result (blocks current thread, but doesn't block Rust I/O threads)
          ArrowReader reader = future.get(10, TimeUnit.SECONDS);
          assertNotNull(reader);

          // Read all batches
          int rowCount = 0;
          while (reader.loadNextBatch()) {
            VectorSchemaRoot root = reader.getVectorSchemaRoot();
            rowCount += root.getRowCount();
          }

          assertEquals(totalRows, rowCount, "Should read all rows");
          reader.close();
        }
      }
    }
  }

  /**
   * Example 2: Async scan with filter.
   *
   * <p>Shows how to use async scanner with SQL-like filters.
   */
  @Test
  void testAsyncScanWithFilter(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("async_scanner_filter").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();

      try (Dataset dataset = testDataset.write(1, 40)) {
        // Scan with filter - only rows where id < 20
        ScanOptions options = new ScanOptions.Builder().filter("id < 20").build();

        try (AsyncScanner scanner = AsyncScanner.create(dataset, options, allocator)) {
          CompletableFuture<ArrowReader> future = scanner.scanBatchesAsync();

          ArrowReader reader = future.get(10, TimeUnit.SECONDS);
          int rowCount = 0;
          while (reader.loadNextBatch()) {
            VectorSchemaRoot root = reader.getVectorSchemaRoot();
            rowCount += root.getRowCount();
          }

          assertEquals(20, rowCount, "Should read only filtered rows");
          reader.close();
        }
      }
    }
  }

  /**
   * Example 3: Multiple concurrent async scans.
   *
   * <p>Shows how to run multiple scans in parallel without blocking threads. This is the key
   * benefit for query engines like Presto/Trino.
   */
  @Test
  void testConcurrentAsyncScans(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("async_scanner_concurrent").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 100;

      try (Dataset dataset = testDataset.write(1, totalRows)) {
        // Create 5 concurrent scans with different filters
        List<CompletableFuture<Integer>> futures = new ArrayList<>();

        for (int i = 0; i < 5; i++) {
          final int rangeStart = i * 20;
          final int rangeEnd = rangeStart + 20;
          String filter = String.format("id >= %d AND id < %d", rangeStart, rangeEnd);

          ScanOptions options = new ScanOptions.Builder().filter(filter).build();

          AsyncScanner scanner = AsyncScanner.create(dataset, options, allocator);

          // Chain async operations: scan -> read -> count rows -> cleanup
          CompletableFuture<Integer> future =
              scanner
                  .scanBatchesAsync()
                  .thenApply(
                      reader -> {
                        try {
                          int count = 0;
                          while (reader.loadNextBatch()) {
                            count += reader.getVectorSchemaRoot().getRowCount();
                          }
                          reader.close();
                          scanner.close();
                          return count;
                        } catch (Exception e) {
                          throw new RuntimeException(e);
                        }
                      });

          futures.add(future);
        }

        // Wait for all scans to complete
        CompletableFuture<Void> allDone =
            CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]));
        allDone.get(30, TimeUnit.SECONDS);

        // Verify each scan read the expected number of rows
        for (CompletableFuture<Integer> future : futures) {
          assertEquals(20, future.get(), "Each range should have 20 rows");
        }
      }
    }
  }

  /**
   * Example 4: Async scan with error handling.
   *
   * <p>Shows how to handle errors in async operations.
   */
  @Test
  void testAsyncScanErrorHandling(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("async_scanner_error").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();

      try (Dataset dataset = testDataset.write(1, 40)) {
        ScanOptions options = new ScanOptions.Builder().build();

        try (AsyncScanner scanner = AsyncScanner.create(dataset, options, allocator)) {
          CompletableFuture<ArrowReader> future =
              scanner
                  .scanBatchesAsync()
                  .whenComplete(
                      (reader, error) -> {
                        if (error != null) {
                          // Handle error
                          System.err.println("Scan failed: " + error.getMessage());
                        } else {
                          // Process successful result
                          assertNotNull(reader);
                        }
                      });

          ArrowReader reader = future.get(10, TimeUnit.SECONDS);
          assertNotNull(reader);
          reader.close();
        }
      }
    }
  }

  /**
   * Example 5: Async scan with projection (column selection).
   *
   * <p>Shows how to select specific columns for better performance.
   */
  @Test
  void testAsyncScanWithProjection(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("async_scanner_projection").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();

      try (Dataset dataset = testDataset.write(1, 40)) {
        // Select only "id" column
        ScanOptions options = new ScanOptions.Builder().columns(List.of("id")).build();

        try (AsyncScanner scanner = AsyncScanner.create(dataset, options, allocator)) {
          CompletableFuture<ArrowReader> future = scanner.scanBatchesAsync();

          ArrowReader reader = future.get(10, TimeUnit.SECONDS);

          // Verify schema has only one column
          assertEquals(1, reader.getVectorSchemaRoot().getFieldVectors().size());
          assertEquals("id", reader.getVectorSchemaRoot().getVector(0).getName());

          reader.close();
        }
      }
    }
  }

  /**
   * Example 6: Using thenCompose for sequential async operations.
   *
   * <p>Shows how to chain multiple async operations together.
   */
  @Test
  void testAsyncChaining(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("async_scanner_chaining").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();

      try (Dataset dataset = testDataset.write(1, 40)) {
        ScanOptions options = new ScanOptions.Builder().build();

        try (AsyncScanner scanner = AsyncScanner.create(dataset, options, allocator)) {
          // Chain operations: scan -> read first batch -> extract values
          CompletableFuture<List<Integer>> future =
              scanner
                  .scanBatchesAsync()
                  .thenApply(
                      reader -> {
                        try {
                          List<Integer> values = new ArrayList<>();
                          if (reader.loadNextBatch()) {
                            VectorSchemaRoot root = reader.getVectorSchemaRoot();
                            IntVector idVector = (IntVector) root.getVector("id");
                            for (int i = 0; i < root.getRowCount(); i++) {
                              values.add(idVector.get(i));
                            }
                          }
                          reader.close();
                          return values;
                        } catch (Exception e) {
                          throw new RuntimeException(e);
                        }
                      });

          List<Integer> values = future.get(10, TimeUnit.SECONDS);
          assertTrue(values.size() > 0, "Should read some values");
        }
      }
    }
  }
}
