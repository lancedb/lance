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
package org.lance.operation;

import org.lance.CommitBuilder;
import org.lance.Dataset;
import org.lance.TestUtils;
import org.lance.Transaction;

import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

public class UpdateConfigTest extends OperationTestBase {

  @Test
  void testUpdateConfig(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testUpdateConfig").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // Test 1: Update configuration values using configUpdates
      Map<String, String> configValues = new HashMap<>();
      configValues.put("key1", "value1");
      configValues.put("key2", "value2");

      UpdateMap configUpdates = UpdateMap.builder().updates(configValues).replace(false).build();

      try (Transaction txn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(UpdateConfig.builder().configUpdates(configUpdates).build())
              .build()) {
        try (Dataset updatedDataset = new CommitBuilder(dataset).execute(txn)) {
          assertEquals(2, updatedDataset.version());
          assertEquals("value1", updatedDataset.getConfig().get("key1"));
          assertEquals("value2", updatedDataset.getConfig().get("key2"));

          // Test 2: Delete configuration key using configUpdates with null value
          Map<String, String> deleteUpdates = new HashMap<>();
          deleteUpdates.put("key1", null); // null value means delete

          UpdateMap configDeleteUpdates =
              UpdateMap.builder().updates(deleteUpdates).replace(false).build();

          try (Transaction txn2 =
              new Transaction.Builder()
                  .readVersion(updatedDataset.version())
                  .operation(UpdateConfig.builder().configUpdates(configDeleteUpdates).build())
                  .build()) {
            try (Dataset updatedDataset2 = new CommitBuilder(updatedDataset).execute(txn2)) {
              assertEquals(3, updatedDataset2.version());
              assertNull(updatedDataset2.getConfig().get("key1"));
              assertEquals("value2", updatedDataset2.getConfig().get("key2"));

              // Test 3: Update schema metadata using schemaMetadataUpdates
              Map<String, String> schemaMetadataMap = new HashMap<>();
              schemaMetadataMap.put("schema_key1", "schema_value1");
              schemaMetadataMap.put("schema_key2", "schema_value2");

              UpdateMap schemaMetadataUpdates =
                  UpdateMap.builder().updates(schemaMetadataMap).replace(false).build();

              try (Transaction txn3 =
                  new Transaction.Builder()
                      .readVersion(updatedDataset2.version())
                      .operation(
                          UpdateConfig.builder()
                              .schemaMetadataUpdates(schemaMetadataUpdates)
                              .build())
                      .build()) {
                try (Dataset updatedDataset3 = new CommitBuilder(updatedDataset2).execute(txn3)) {
                  assertEquals(4, updatedDataset3.version());
                  assertEquals(
                      "schema_value1",
                      updatedDataset3.getLanceSchema().metadata().get("schema_key1"));
                  assertEquals(
                      "schema_value2",
                      updatedDataset3.getLanceSchema().metadata().get("schema_key2"));

                  // Test 4: Update field metadata using fieldMetadataUpdates
                  Map<Integer, UpdateMap> fieldMetadataUpdates = new HashMap<>();

                  Map<String, String> field0Updates = new HashMap<>();
                  field0Updates.put("field0_key1", "field0_value1");
                  UpdateMap field0UpdateMap =
                      UpdateMap.builder().updates(field0Updates).replace(false).build();

                  Map<String, String> field1Updates = new HashMap<>();
                  field1Updates.put("field1_key1", "field1_value1");
                  field1Updates.put("field1_key2", "field1_value2");
                  UpdateMap field1UpdateMap =
                      UpdateMap.builder().updates(field1Updates).replace(false).build();

                  fieldMetadataUpdates.put(0, field0UpdateMap);
                  fieldMetadataUpdates.put(1, field1UpdateMap);

                  try (Transaction txn4 =
                      new Transaction.Builder()
                          .readVersion(updatedDataset3.version())
                          .operation(
                              UpdateConfig.builder()
                                  .fieldMetadataUpdates(fieldMetadataUpdates)
                                  .build())
                          .build()) {
                    try (Dataset updatedDataset4 =
                        new CommitBuilder(updatedDataset3).execute(txn4)) {
                      assertEquals(5, updatedDataset4.version());

                      // Verify field metadata for field 0
                      Map<String, String> fieldMetadata0 =
                          updatedDataset4.getLanceSchema().fields().get(0).getMetadata();
                      assertEquals("field0_value1", fieldMetadata0.get("field0_key1"));

                      // Verify field metadata for field 1
                      Map<String, String> field1Result =
                          updatedDataset4.getLanceSchema().fields().get(1).getMetadata();
                      assertEquals("field1_value1", field1Result.get("field1_key1"));
                      assertEquals("field1_value2", field1Result.get("field1_key2"));
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
