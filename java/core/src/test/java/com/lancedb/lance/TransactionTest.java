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
import com.lancedb.lance.operation.Append;
import com.lancedb.lance.operation.Merge;
import com.lancedb.lance.operation.Overwrite;
import com.lancedb.lance.operation.Project;

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TransactionTest {
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
  void testProjection(@TempDir Path tempDir) {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      assertEquals(testDataset.getSchema(), dataset.getSchema());
      List<Field> fieldList = new ArrayList<>(testDataset.getSchema().getFields());
      Collections.reverse(fieldList);
      Transaction txn1 =
          dataset
              .newTransactionBuilder()
              .operation(Project.builder().schema(new Schema(fieldList)).build())
              .build();
      try (Dataset committedDataset = txn1.commit()) {
        assertEquals(1, txn1.readVersion());
        assertEquals(1, dataset.version());
        assertEquals(2, committedDataset.version());
        assertEquals(new Schema(fieldList), committedDataset.getSchema());
        fieldList.remove(1);
        Transaction txn2 =
            committedDataset
                .newTransactionBuilder()
                .operation(Project.builder().schema(new Schema(fieldList)).build())
                .build();
        try (Dataset committedDataset2 = txn2.commit()) {
          assertEquals(2, txn2.readVersion());
          assertEquals(2, committedDataset.version());
          assertEquals(3, committedDataset2.version());
          assertEquals(new Schema(fieldList), committedDataset2.getSchema());
          assertEquals(txn1, committedDataset.readTransaction().orElse(null));
          assertEquals(txn2, committedDataset2.readTransaction().orElse(null));
        }
      }
    }
  }

  @Test
  void testAppend(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testAppend").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();
      // Commit fragment
      int rowCount = 20;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(rowCount);
      Transaction transaction =
          dataset
              .newTransactionBuilder()
              .operation(
                  Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
              .transactionProperties(Collections.singletonMap("key", "value"))
              .build();
      assertEquals("value", transaction.transactionProperties().get("key"));
      try (Dataset dataset = transaction.commit()) {
        assertEquals(2, dataset.version());
        assertEquals(2, dataset.latestVersion());
        assertEquals(rowCount, dataset.countRows());
        assertThrows(
            IllegalArgumentException.class,
            () ->
                dataset
                    .newTransactionBuilder()
                    .operation(Append.builder().fragments(new ArrayList<>()).build())
                    .build()
                    .commit()
                    .close());
        assertEquals(transaction, dataset.readTransaction().orElse(null));
      }
    }
  }

  @Test
  void testOverwrite(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testOverwrite").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // Commit fragment
      int rowCount = 20;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(rowCount);
      Transaction transaction =
          dataset
              .newTransactionBuilder()
              .operation(
                  Overwrite.builder()
                      .fragments(Collections.singletonList(fragmentMeta))
                      .schema(testDataset.getSchema())
                      .build())
              .build();
      try (Dataset dataset = transaction.commit()) {
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
      transaction =
          dataset
              .newTransactionBuilder()
              .operation(
                  Overwrite.builder()
                      .fragments(Collections.singletonList(fragmentMeta))
                      .schema(testDataset.getSchema())
                      .configUpsertValues(Collections.singletonMap("config_key", "config_value"))
                      .build())
              .transactionProperties(Collections.singletonMap("key", "value"))
              .build();
      assertEquals("value", transaction.transactionProperties().get("key"));
      try (Dataset dataset = transaction.commit()) {
        assertEquals(3, dataset.version());
        assertEquals(3, dataset.latestVersion());
        assertEquals(rowCount, dataset.countRows());
        assertEquals("config_value", dataset.getConfig().get("config_key"));
        Fragment fragment = dataset.getFragments().get(0);

        try (LanceScanner scanner = fragment.newScan()) {
          Schema schemaRes = scanner.schema();
          assertEquals(testDataset.getSchema(), schemaRes);
        }
        assertEquals(transaction, dataset.readTransaction().orElse(null));
      }
    }
  }

  @Test
  void testMerge(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testMerge").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      int rowCount = 20;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(rowCount);
      Transaction transaction =
          dataset
              .newTransactionBuilder()
              .operation(
                  Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
              .build();
      try (Dataset dataset = transaction.commit()) {
        assertEquals(2, dataset.version());
        assertEquals(rowCount, dataset.countRows());

        Schema newSchema = testDataset.getSchema();
        FragmentMetadata newFragmentMeta = testDataset.createNewFragment(rowCount);

        Transaction mergeTransaction =
            dataset
                .newTransactionBuilder()
                .operation(
                    Merge.builder()
                        .fragments(Collections.singletonList(newFragmentMeta))
                        .schema(newSchema)
                        .build())
                .transactionProperties(Collections.singletonMap("key", "value"))
                .build();
        assertEquals("value", mergeTransaction.transactionProperties().get("key"));
        try (Dataset mergedDataset = mergeTransaction.commit()) {
          assertEquals(3, mergedDataset.version());
          assertEquals(3, mergedDataset.latestVersion());
          assertEquals(rowCount, mergedDataset.countRows());

          assertEquals(newSchema, mergedDataset.getSchema());

          Fragment fragment = mergedDataset.getFragments().get(0);
          try (LanceScanner scanner = fragment.newScan()) {
            Schema schemaRes = scanner.schema();
            assertEquals(newSchema, schemaRes);
          }

          assertEquals(mergeTransaction, mergedDataset.readTransaction().orElse(null));
        }
      }
    }
  }
}
