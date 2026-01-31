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

import org.apache.arrow.memory.RootAllocator;
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
  public void testTransaction(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testTransaction").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        FragmentMetadata fragmentMeta = testDataset.createNewFragment(20);

        Map<String, String> properties = new HashMap<>();
        properties.put("transactionType", "APPEND");
        properties.put("createdBy", "testUser");
        Transaction appendTxn =
            dataset
                .newTransactionBuilder()
                .operation(
                    Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
                .transactionProperties(properties)
                .build();
        try (Dataset committedDataset = appendTxn.commit()) {
          assertEquals(2, committedDataset.version());
          assertEquals(2, committedDataset.latestVersion());
          assertEquals(20, committedDataset.countRows());
          assertEquals(dataset.version(), appendTxn.readVersion());
          assertNotNull(appendTxn.uuid());

          // Verify transaction properties
          Map<String, String> txnProps = appendTxn.transactionProperties().orElse(new HashMap<>());
          assertEquals("APPEND", txnProps.get("transactionType"));
          assertEquals("testUser", txnProps.get("createdBy"));
        }
      }
    }
  }

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
}
