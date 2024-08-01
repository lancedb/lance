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

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Path;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
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
  void testCreateDirNotExist() throws IOException, URISyntaxException {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (BufferAllocator allocator = new RootAllocator()) {
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
  void testOpenNonExist() throws IOException, URISyntaxException {
    String datasetPath = tempDir.resolve("non_exist").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      assertThrows(IllegalArgumentException.class, () -> {
        Dataset.open(datasetPath, allocator);
      });
    }
  }

  @Test
  void testCreateExist() throws IOException, URISyntaxException {
    String datasetPath = tempDir.resolve("create_exist").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      assertThrows(IllegalArgumentException.class, () -> {
        testDataset.createEmptyDataset();
      });
    }
  }

  @Test
  void testCommitConflict() {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        assertEquals(1, dataset.version());
        assertEquals(1, dataset.latestVersion());
        assertThrows(IllegalArgumentException.class, () -> {
          testDataset.write(0, 5);
        });
      }
    }
  }

  @Test
  void testGetSchemaWithClosedDataset() {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
      Dataset dataset = testDataset.createEmptyDataset();
      dataset.close();
      assertThrows(RuntimeException.class, dataset::getSchema);
    }
  }
}
