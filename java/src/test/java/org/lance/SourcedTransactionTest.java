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

import org.lance.operation.Append;

import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class SourcedTransactionTest {

  @Test
  public void testSourcedTransaction(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testSourcedTransaction").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        FragmentMetadata fragmentMeta = testDataset.createNewFragment(20);

        Map<String, String> properties = new HashMap<>();
        properties.put("transactionType", "APPEND");
        properties.put("createdBy", "testUser");
        try (SourcedTransaction appendTxn =
            dataset
                .newTransactionBuilder()
                .operation(
                    Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
                .transactionProperties(properties)
                .build()) {
          try (Dataset committedDataset = appendTxn.commit()) {
            assertEquals(2, committedDataset.version());
            assertEquals(2, committedDataset.latestVersion());
            assertEquals(20, committedDataset.countRows());
            assertEquals(dataset.version(), appendTxn.readVersion());
            assertNotNull(appendTxn.uuid());

            // Verify transaction properties
            Map<String, String> txnProps =
                appendTxn.transactionProperties().orElse(new HashMap<>());
            assertEquals("APPEND", txnProps.get("transactionType"));
            assertEquals("testUser", txnProps.get("createdBy"));
          }
        }
      }
    }
  }

  @Test
  public void testTag(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testTag").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        FragmentMetadata fragmentMeta = testDataset.createNewFragment(10);

        try (SourcedTransaction txn =
            dataset
                .newTransactionBuilder()
                .tag("release-v2")
                .operation(
                    Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
                .build()) {
          assertEquals("release-v2", txn.tag().orElse(null));
          assertEquals("release-v2", txn.transaction().tag().orElse(null));

          try (Dataset committed = txn.commit()) {
            Transaction readTx = committed.readTransaction().orElse(null);
            assertNotNull(readTx);
            assertEquals("release-v2", readTx.tag().orElse(null));
          }
        }
      }
    }
  }

  @Test
  public void testReadVersionDefaultsToDatasetVersion(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testReadVersionDefault").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        FragmentMetadata fragmentMeta = testDataset.createNewFragment(10);

        // Do not set readVersion explicitly — it should default to dataset.version()
        try (SourcedTransaction txn =
            dataset
                .newTransactionBuilder()
                .operation(
                    Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
                .build()) {
          assertEquals(dataset.version(), txn.readVersion());

          try (Dataset committed = txn.commit()) {
            assertTrue(committed.version() > dataset.version());
          }
        }
      }
    }
  }
}
