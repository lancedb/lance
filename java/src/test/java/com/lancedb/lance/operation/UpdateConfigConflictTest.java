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
package com.lancedb.lance.operation;

import com.lancedb.lance.Dataset;
import com.lancedb.lance.TestUtils;
import com.lancedb.lance.Transaction;

import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

public class UpdateConfigConflictTest extends OperationTestBase {

  @Test
  void testSchemaMetadataConflictDetection(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testSchemaConflict").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // Create two concurrent transactions that both modify schema metadata
      Map<String, String> schemaUpdates1 = new HashMap<>();
      schemaUpdates1.put("modified_by", "user1");
      schemaUpdates1.put("timestamp", "2024-01-01");

      Map<String, String> schemaUpdates2 = new HashMap<>();
      schemaUpdates2.put("modified_by", "user2");
      schemaUpdates2.put("description", "Updated by user2");

      UpdateMap schemaUpdateMap1 =
          UpdateMap.builder().updates(schemaUpdates1).replace(false).build();
      UpdateMap schemaUpdateMap2 =
          UpdateMap.builder().updates(schemaUpdates2).replace(false).build();

      UpdateConfig updateConfig1 =
          UpdateConfig.builder().schemaMetadataUpdates(schemaUpdateMap1).build();
      UpdateConfig updateConfig2 =
          UpdateConfig.builder().schemaMetadataUpdates(schemaUpdateMap2).build();

      // Create concurrent transactions from the same base version
      long baseVersion = dataset.version();
      Transaction transaction1 = dataset.newTransactionBuilder().operation(updateConfig1).build();
      Transaction transaction2 = dataset.newTransactionBuilder().operation(updateConfig2).build();

      // First transaction should succeed
      Dataset dataset1 = transaction1.commit();
      assertEquals(baseVersion + 1, dataset1.version());

      // Second transaction should detect conflict and succeed with latest version
      // Note: The exact conflict resolution behavior depends on Lance implementation
      // This test documents the expected behavior for schema metadata conflicts
      Dataset dataset2 = transaction2.commit();
      assertTrue(dataset2.version() > dataset1.version());
    }
  }

  @Test
  void testFieldMetadataConflictDetection(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testFieldConflict").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // Create two concurrent transactions that modify the same field metadata
      Map<String, String> field0Updates1 = new HashMap<>();
      field0Updates1.put("encoding", "utf8");
      field0Updates1.put("indexed", "true");

      Map<String, String> field0Updates2 = new HashMap<>();
      field0Updates2.put("encoding", "utf16"); // Conflicting value
      field0Updates2.put("nullable", "false");

      UpdateMap field0UpdateMap1 =
          UpdateMap.builder().updates(field0Updates1).replace(false).build();
      UpdateMap field0UpdateMap2 =
          UpdateMap.builder().updates(field0Updates2).replace(false).build();

      Map<Integer, UpdateMap> fieldUpdates1 = new HashMap<>();
      fieldUpdates1.put(0, field0UpdateMap1);

      Map<Integer, UpdateMap> fieldUpdates2 = new HashMap<>();
      fieldUpdates2.put(0, field0UpdateMap2);

      UpdateConfig updateConfig1 =
          UpdateConfig.builder().fieldMetadataUpdates(fieldUpdates1).build();
      UpdateConfig updateConfig2 =
          UpdateConfig.builder().fieldMetadataUpdates(fieldUpdates2).build();

      // Create concurrent transactions from the same base version
      long baseVersion = dataset.version();
      Transaction transaction1 = dataset.newTransactionBuilder().operation(updateConfig1).build();
      Transaction transaction2 = dataset.newTransactionBuilder().operation(updateConfig2).build();

      // First transaction should succeed
      Dataset dataset1 = transaction1.commit();
      assertEquals(baseVersion + 1, dataset1.version());

      // Second transaction should handle the conflict appropriately
      Dataset dataset2 = transaction2.commit();
      assertTrue(dataset2.version() > dataset1.version());
    }
  }

  @Test
  void testNonConflictingConcurrentUpdates(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testNonConflicting").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // Create two transactions that modify different metadata types (should not conflict)
      Map<String, String> configUpdates = new HashMap<>();
      configUpdates.put("timeout", "30s");

      Map<String, String> tableUpdates = new HashMap<>();
      tableUpdates.put("description", "Test dataset");

      UpdateMap configUpdateMap = UpdateMap.builder().updates(configUpdates).replace(false).build();
      UpdateMap tableUpdateMap = UpdateMap.builder().updates(tableUpdates).replace(false).build();

      UpdateConfig updateConfig1 = UpdateConfig.builder().configUpdates(configUpdateMap).build();
      UpdateConfig updateConfig2 =
          UpdateConfig.builder().tableMetadataUpdates(tableUpdateMap).build();

      // Create concurrent transactions from the same base version
      long baseVersion = dataset.version();
      Transaction transaction1 = dataset.newTransactionBuilder().operation(updateConfig1).build();
      Transaction transaction2 = dataset.newTransactionBuilder().operation(updateConfig2).build();

      // Both transactions should succeed since they modify different metadata types
      Dataset dataset1 = transaction1.commit();
      assertEquals(baseVersion + 1, dataset1.version());

      Dataset dataset2 = transaction2.commit();
      assertEquals(baseVersion + 2, dataset2.version());

      // Verify both updates are present
      assertTrue(dataset2.getConfig().containsKey("timeout"));
      assertEquals("30s", dataset2.getConfig().get("timeout"));
      // Note: table metadata verification would require additional API methods
    }
  }

  @Test
  void testConfigKeyConflictResolution(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testConfigConflict").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // Set initial config value
      Map<String, String> initialConfig = new HashMap<>();
      initialConfig.put("shared_key", "initial_value");
      UpdateMap initialUpdateMap =
          UpdateMap.builder().updates(initialConfig).replace(false).build();
      UpdateConfig initialUpdate = UpdateConfig.builder().configUpdates(initialUpdateMap).build();

      Transaction initialTransaction =
          dataset.newTransactionBuilder().operation(initialUpdate).build();
      dataset = initialTransaction.commit();

      // Create two concurrent transactions that modify the same config key
      Map<String, String> configUpdates1 = new HashMap<>();
      configUpdates1.put("shared_key", "value_from_user1");
      configUpdates1.put("user1_key", "user1_data");

      Map<String, String> configUpdates2 = new HashMap<>();
      configUpdates2.put("shared_key", "value_from_user2");
      configUpdates2.put("user2_key", "user2_data");

      UpdateMap configUpdateMap1 =
          UpdateMap.builder().updates(configUpdates1).replace(false).build();
      UpdateMap configUpdateMap2 =
          UpdateMap.builder().updates(configUpdates2).replace(false).build();

      UpdateConfig updateConfig1 = UpdateConfig.builder().configUpdates(configUpdateMap1).build();
      UpdateConfig updateConfig2 = UpdateConfig.builder().configUpdates(configUpdateMap2).build();

      // Create concurrent transactions from the same base version
      long baseVersion = dataset.version();
      Transaction transaction1 = dataset.newTransactionBuilder().operation(updateConfig1).build();
      Transaction transaction2 = dataset.newTransactionBuilder().operation(updateConfig2).build();

      // First transaction should succeed
      Dataset dataset1 = transaction1.commit();
      assertEquals(baseVersion + 1, dataset1.version());
      assertEquals("value_from_user1", dataset1.getConfig().get("shared_key"));
      assertEquals("user1_data", dataset1.getConfig().get("user1_key"));

      // Second transaction should handle the conflict
      // The exact behavior depends on Lance's conflict resolution strategy
      Dataset dataset2 = transaction2.commit();
      assertTrue(dataset2.version() > dataset1.version());

      // Verify that the final state contains updates from both transactions where possible
      Map<String, String> finalConfig = dataset2.getConfig();
      assertTrue(finalConfig.containsKey("shared_key")); // One of the values should win
      assertTrue(finalConfig.containsKey("user1_key")); // Non-conflicting keys should be preserved
      assertTrue(finalConfig.containsKey("user2_key")); // Non-conflicting keys should be applied
    }
  }

  @Test
  void testMultipleFieldConflicts(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testMultipleFields").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // Create transaction that modifies multiple fields simultaneously
      Map<String, String> field0Updates = new HashMap<>();
      field0Updates.put("type", "string");
      field0Updates.put("max_length", "100");

      Map<String, String> field1Updates = new HashMap<>();
      field1Updates.put("type", "integer");
      field1Updates.put("min_value", "0");

      UpdateMap field0UpdateMap = UpdateMap.builder().updates(field0Updates).replace(false).build();
      UpdateMap field1UpdateMap = UpdateMap.builder().updates(field1Updates).replace(false).build();

      Map<Integer, UpdateMap> fieldUpdates1 = new HashMap<>();
      fieldUpdates1.put(0, field0UpdateMap);
      fieldUpdates1.put(1, field1UpdateMap);

      // Second transaction modifies different fields
      Map<String, String> field2Updates = new HashMap<>();
      field2Updates.put("type", "float");
      field2Updates.put("precision", "64");

      UpdateMap field2UpdateMap = UpdateMap.builder().updates(field2Updates).replace(false).build();

      Map<Integer, UpdateMap> fieldUpdates2 = new HashMap<>();
      fieldUpdates2.put(2, field2UpdateMap); // Different field, should not conflict
      Map<String, String> field0ConflictUpdates = new HashMap<>();
      field0ConflictUpdates.put("nullable", "true");
      fieldUpdates2.put(
          0,
          UpdateMap.builder()
              .updates(field0ConflictUpdates)
              .replace(false)
              .build()); // Same field, potential conflict

      UpdateConfig updateConfig1 =
          UpdateConfig.builder().fieldMetadataUpdates(fieldUpdates1).build();
      UpdateConfig updateConfig2 =
          UpdateConfig.builder().fieldMetadataUpdates(fieldUpdates2).build();

      // Create concurrent transactions
      long baseVersion = dataset.version();
      Transaction transaction1 = dataset.newTransactionBuilder().operation(updateConfig1).build();
      Transaction transaction2 = dataset.newTransactionBuilder().operation(updateConfig2).build();

      // Both should succeed, testing conflict resolution for overlapping field metadata
      Dataset dataset1 = transaction1.commit();
      assertEquals(baseVersion + 1, dataset1.version());

      Dataset dataset2 = transaction2.commit();
      assertTrue(dataset2.version() > dataset1.version());
    }
  }
}
