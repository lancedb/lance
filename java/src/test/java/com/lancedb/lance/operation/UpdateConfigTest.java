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
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
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

      // Test 1: Update configuration values using upsertValues
      Map<String, String> upsertValues = new HashMap<>();
      upsertValues.put("key1", "value1");
      upsertValues.put("key2", "value2");

      Transaction transaction =
          dataset
              .newTransactionBuilder()
              .operation(UpdateConfig.builder().upsertValues(upsertValues).build())
              .build();
      try (Dataset updatedDataset = transaction.commit()) {
        assertEquals(2, updatedDataset.version());
        assertEquals("value1", updatedDataset.getConfig().get("key1"));
        assertEquals("value2", updatedDataset.getConfig().get("key2"));

        // Test 2: Delete configuration keys using deleteKeys
        List<String> deleteKeys = Collections.singletonList("key1");
        transaction =
            updatedDataset
                .newTransactionBuilder()
                .operation(UpdateConfig.builder().deleteKeys(deleteKeys).build())
                .build();
        try (Dataset updatedDataset2 = transaction.commit()) {
          assertEquals(3, updatedDataset2.version());
          assertNull(updatedDataset2.getConfig().get("key1"));
          assertEquals("value2", updatedDataset2.getConfig().get("key2"));

          // Test 3: Update schema metadata using schemaMetadata
          Map<String, String> schemaMetadata = new HashMap<>();
          schemaMetadata.put("schema_key1", "schema_value1");
          schemaMetadata.put("schema_key2", "schema_value2");

          transaction =
              updatedDataset2
                  .newTransactionBuilder()
                  .operation(UpdateConfig.builder().schemaMetadata(schemaMetadata).build())
                  .build();
          try (Dataset updatedDataset3 = transaction.commit()) {
            assertEquals(4, updatedDataset3.version());
            assertEquals(
                "schema_value1", updatedDataset3.getLanceSchema().metadata().get("schema_key1"));
            assertEquals(
                "schema_value2", updatedDataset3.getLanceSchema().metadata().get("schema_key2"));

            // Test 4: Update field metadata using fieldMetadata
            Map<Integer, Map<String, String>> fieldMetadata = new HashMap<>();
            Map<String, String> field0Metadata = new HashMap<>();
            field0Metadata.put("field0_key1", "field0_value1");

            Map<String, String> field1Metadata = new HashMap<>();
            field1Metadata.put("field1_key1", "field1_value1");
            field1Metadata.put("field1_key2", "field1_value2");

            fieldMetadata.put(0, field0Metadata);
            fieldMetadata.put(1, field1Metadata);

            transaction =
                updatedDataset3
                    .newTransactionBuilder()
                    .operation(UpdateConfig.builder().fieldMetadata(fieldMetadata).build())
                    .build();
            try (Dataset updatedDataset4 = transaction.commit()) {
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
