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

import org.lance.index.IndexOptions;
import org.lance.index.IndexParams;
import org.lance.index.IndexType;
import org.lance.index.scalar.ScalarIndexParams;
import org.lance.operation.Append;
import org.lance.operation.CreateIndex;
import org.lance.operation.Overwrite;

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TransactionTest {

  @Test
  public void testReadTransactionCreateIndex(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("read_transaction_create_index").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      try (Dataset dataset = testDataset.createEmptyDataset()) {
        assertEquals(1, dataset.version());
      }

      try (Dataset dataset = testDataset.write(1, 10)) {
        ScalarIndexParams scalarParams = ScalarIndexParams.create("btree", "{\"zone_size\": 2048}");
        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(scalarParams).build();

        dataset.createIndex(
            IndexOptions.builder(Collections.singletonList("id"), IndexType.BTREE, indexParams)
                .withIndexName("btree_id_index")
                .build());

        assertTrue(
            dataset.listIndexes().contains("btree_id_index"),
            "Expected 'btree_id_index' to be created");

        Transaction readTx = dataset.readTransaction().orElse(null);
        assertNotNull(readTx, "readTransaction() should return a transaction for CreateIndex");
        assertEquals("CreateIndex", readTx.operation().name());

        assertInstanceOf(CreateIndex.class, readTx.operation());
        CreateIndex op = (CreateIndex) readTx.operation();
        assertFalse(op.getNewIndices().isEmpty(), "newIndices should not be empty for CreateIndex");
        assertTrue(
            op.getRemovedIndices().isEmpty(), "removedIndices should be empty for CreateIndex");
        assertEquals("btree_id_index", (op.getNewIndices().get(0).name()));
      }
    }
  }

  @Test
  public void testCommitToUri(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testCommitToUri").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      Schema schema = testDataset.getSchema();

      // Create fragments at the dataset path
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(20);

      // Build a transaction targeting a URI (no existing dataset)
      try (Transaction txn =
          new Transaction.Builder()
              .operation(
                  Overwrite.builder()
                      .fragments(Collections.singletonList(fragmentMeta))
                      .schema(schema)
                      .build())
              .build()) {
        try (Dataset committedDataset = new CommitBuilder(datasetPath, allocator).execute(txn)) {
          assertEquals(1, committedDataset.version());
          assertEquals(20, committedDataset.countRows());
        }
      }
    }
  }

  @Test
  public void testTagRoundTrip(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testTagRoundTrip").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        FragmentMetadata fragmentMeta = testDataset.createNewFragment(10);

        try (Transaction txn =
            new Transaction.Builder()
                .readVersion(dataset.version())
                .tag("v1.0")
                .operation(
                    Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
                .build()) {
          assertEquals("v1.0", txn.tag().orElse(null));

          try (Dataset committed = new CommitBuilder(dataset).execute(txn)) {
            Transaction readTx = committed.readTransaction().orElse(null);
            assertNotNull(readTx);
            assertEquals("v1.0", readTx.tag().orElse(null));
          }
        }
      }
    }
  }

  @Test
  public void testTransactionPropertiesRoundTrip(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testTransactionPropertiesRoundTrip").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        FragmentMetadata fragmentMeta = testDataset.createNewFragment(10);

        Map<String, String> properties = new HashMap<>();
        properties.put("source", "ingestion-pipeline");
        properties.put("batchId", "42");

        try (Transaction txn =
            new Transaction.Builder()
                .readVersion(dataset.version())
                .transactionProperties(properties)
                .operation(
                    Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
                .build()) {
          try (Dataset committed = new CommitBuilder(dataset).execute(txn)) {
            Transaction readTx = committed.readTransaction().orElse(null);
            assertNotNull(readTx);
            Map<String, String> readProps = readTx.transactionProperties().orElse(null);
            assertNotNull(readProps);
            assertEquals("ingestion-pipeline", readProps.get("source"));
            assertEquals("42", readProps.get("batchId"));
          }
        }
      }
    }
  }

  @Test
  public void testCustomUuid(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testCustomUuid").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        FragmentMetadata fragmentMeta = testDataset.createNewFragment(10);

        String customUuid = "custom-uuid-12345";
        try (Transaction txn =
            new Transaction.Builder()
                .readVersion(dataset.version())
                .uuid(customUuid)
                .operation(
                    Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
                .build()) {
          assertEquals(customUuid, txn.uuid());

          try (Dataset committed = new CommitBuilder(dataset).execute(txn)) {
            Transaction readTx = committed.readTransaction().orElse(null);
            assertNotNull(readTx);
            assertEquals(customUuid, readTx.uuid());
          }
        }
      }
    }
  }
}
