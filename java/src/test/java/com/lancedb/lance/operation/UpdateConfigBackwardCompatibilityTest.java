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

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

public class UpdateConfigBackwardCompatibilityTest {

  @Test
  @SuppressWarnings("deprecation") // We're testing deprecated methods
  void testDeprecatedUpsertValuesMethod() {
    // Create UpdateConfig using new API
    Map<String, String> updates = new HashMap<>();
    updates.put("key1", "value1");
    updates.put("key2", "value2");
    updates.put("key3", null); // deletion

    UpdateMap configUpdates = UpdateMap.builder().updates(updates).replace(false).build();

    UpdateConfig updateConfig = UpdateConfig.builder().configUpdates(configUpdates).build();

    // Test deprecated upsertValues() method extracts only non-null values
    assertTrue(updateConfig.upsertValues().isPresent());
    Map<String, String> upsertValues = updateConfig.upsertValues().get();
    assertEquals(2, upsertValues.size());
    assertEquals("value1", upsertValues.get("key1"));
    assertEquals("value2", upsertValues.get("key2"));
    assertFalse(upsertValues.containsKey("key3"));
  }

  @Test
  @SuppressWarnings("deprecation") // We're testing deprecated methods
  void testDeprecatedDeleteKeysMethod() {
    // Create UpdateConfig using new API
    Map<String, String> updates = new HashMap<>();
    updates.put("key1", "value1");
    updates.put("delete_key1", null); // deletion
    updates.put("delete_key2", null); // deletion

    UpdateMap configUpdates = UpdateMap.builder().updates(updates).replace(false).build();

    UpdateConfig updateConfig = UpdateConfig.builder().configUpdates(configUpdates).build();

    // Test deprecated deleteKeys() method extracts only null values
    assertTrue(updateConfig.deleteKeys().isPresent());
    List<String> deleteKeys = updateConfig.deleteKeys().get();
    assertEquals(2, deleteKeys.size());
    assertTrue(deleteKeys.contains("delete_key1"));
    assertTrue(deleteKeys.contains("delete_key2"));
    assertFalse(deleteKeys.contains("key1"));
  }

  @Test
  @SuppressWarnings("deprecation") // We're testing deprecated methods
  void testDeprecatedSchemaMetadataMethod() {
    // Create UpdateConfig using new API
    Map<String, String> schemaUpdates = new HashMap<>();
    schemaUpdates.put("schema_key1", "schema_value1");
    schemaUpdates.put("schema_key2", "schema_value2");

    UpdateMap schemaMetadataUpdates =
        UpdateMap.builder().updates(schemaUpdates).replace(false).build();

    UpdateConfig updateConfig =
        UpdateConfig.builder().schemaMetadataUpdates(schemaMetadataUpdates).build();

    // Test deprecated schemaMetadata() method
    assertTrue(updateConfig.schemaMetadata().isPresent());
    Map<String, String> schemaMetadata = updateConfig.schemaMetadata().get();
    assertEquals(2, schemaMetadata.size());
    assertEquals("schema_value1", schemaMetadata.get("schema_key1"));
    assertEquals("schema_value2", schemaMetadata.get("schema_key2"));
  }

  @Test
  @SuppressWarnings("deprecation") // We're testing deprecated methods
  void testDeprecatedFieldMetadataMethod() {
    // Create UpdateConfig using new API
    Map<String, String> field0Updates = new HashMap<>();
    field0Updates.put("field0_key", "field0_value");

    Map<String, String> field1Updates = new HashMap<>();
    field1Updates.put("field1_key", "field1_value");

    UpdateMap field0UpdateMap = UpdateMap.builder().updates(field0Updates).replace(false).build();
    UpdateMap field1UpdateMap = UpdateMap.builder().updates(field1Updates).replace(false).build();

    Map<Integer, UpdateMap> fieldMetadataUpdates = new HashMap<>();
    fieldMetadataUpdates.put(0, field0UpdateMap);
    fieldMetadataUpdates.put(1, field1UpdateMap);

    UpdateConfig updateConfig =
        UpdateConfig.builder().fieldMetadataUpdates(fieldMetadataUpdates).build();

    // Test deprecated fieldMetadata() method
    assertTrue(updateConfig.fieldMetadata().isPresent());
    Map<Integer, Map<String, String>> fieldMetadata = updateConfig.fieldMetadata().get();
    assertEquals(2, fieldMetadata.size());
    assertEquals("field0_value", fieldMetadata.get(0).get("field0_key"));
    assertEquals("field1_value", fieldMetadata.get(1).get("field1_key"));
  }

  @Test
  @SuppressWarnings("deprecation") // We're testing deprecated methods
  void testDeprecatedBuilderUpsertValuesMethod() {
    // Test deprecated builder methods
    Map<String, String> upsertValues = new HashMap<>();
    upsertValues.put("key1", "value1");
    upsertValues.put("key2", "value2");

    UpdateConfig updateConfig = UpdateConfig.builder().upsertValues(upsertValues).build();

    // Verify the new API works with values set by deprecated method
    assertNotNull(updateConfig.configUpdates());
    assertEquals("value1", updateConfig.configUpdates().updates().get("key1"));
    assertEquals("value2", updateConfig.configUpdates().updates().get("key2"));
    assertFalse(updateConfig.configUpdates().replace());
  }

  @Test
  @SuppressWarnings("deprecation") // We're testing deprecated methods
  void testDeprecatedBuilderDeleteKeysMethod() {
    List<String> deleteKeys = Arrays.asList("delete_key1", "delete_key2");

    UpdateConfig updateConfig = UpdateConfig.builder().deleteKeys(deleteKeys).build();

    // Verify the new API shows deletions as null values
    assertNotNull(updateConfig.configUpdates());
    assertNull(updateConfig.configUpdates().updates().get("delete_key1"));
    assertNull(updateConfig.configUpdates().updates().get("delete_key2"));
    assertFalse(updateConfig.configUpdates().replace());
  }

  @Test
  @SuppressWarnings("deprecation") // We're testing deprecated methods
  void testDeprecatedBuilderCombinedUpsertAndDelete() {
    // Test combining upsert and delete operations using deprecated API
    Map<String, String> upsertValues = new HashMap<>();
    upsertValues.put("key1", "value1");
    upsertValues.put("key2", "value2");

    List<String> deleteKeys = Arrays.asList("delete_key1", "delete_key2");

    UpdateConfig updateConfig =
        UpdateConfig.builder().upsertValues(upsertValues).deleteKeys(deleteKeys).build();

    // Verify combined operations work correctly
    assertNotNull(updateConfig.configUpdates());
    Map<String, String> updates = updateConfig.configUpdates().updates();
    assertEquals(4, updates.size());
    assertEquals("value1", updates.get("key1"));
    assertEquals("value2", updates.get("key2"));
    assertNull(updates.get("delete_key1"));
    assertNull(updates.get("delete_key2"));
    assertFalse(updateConfig.configUpdates().replace());
  }

  @Test
  @SuppressWarnings("deprecation") // We're testing deprecated methods
  void testDeprecatedBuilderSchemaMetadataMethod() {
    Map<String, String> schemaMetadata = new HashMap<>();
    schemaMetadata.put("schema_key", "schema_value");

    UpdateConfig updateConfig = UpdateConfig.builder().schemaMetadata(schemaMetadata).build();

    // Verify the new API works with values set by deprecated method
    assertNotNull(updateConfig.schemaMetadataUpdates());
    assertEquals("schema_value", updateConfig.schemaMetadataUpdates().updates().get("schema_key"));
    assertFalse(updateConfig.schemaMetadataUpdates().replace());
  }

  @Test
  @SuppressWarnings("deprecation") // We're testing deprecated methods
  void testDeprecatedBuilderFieldMetadataMethod() {
    Map<String, String> field0Metadata = new HashMap<>();
    field0Metadata.put("field0_key", "field0_value");

    Map<Integer, Map<String, String>> fieldMetadata = new HashMap<>();
    fieldMetadata.put(0, field0Metadata);

    UpdateConfig updateConfig = UpdateConfig.builder().fieldMetadata(fieldMetadata).build();

    // Verify the new API works with values set by deprecated method
    assertNotNull(updateConfig.fieldMetadataUpdates());
    assertEquals(1, updateConfig.fieldMetadataUpdates().size());
    UpdateMap field0UpdateMap = updateConfig.fieldMetadataUpdates().get(0);
    assertNotNull(field0UpdateMap);
    assertEquals("field0_value", field0UpdateMap.updates().get("field0_key"));
    assertFalse(field0UpdateMap.replace());
  }

  @Test
  void testNoRegressionInNewAPI() {
    // Ensure new API still works as expected
    Map<String, String> configUpdates = new HashMap<>();
    configUpdates.put("config_key", "config_value");
    configUpdates.put("delete_key", null);

    UpdateMap configUpdateMap = UpdateMap.builder().updates(configUpdates).replace(false).build();

    Map<String, String> tableUpdates = new HashMap<>();
    tableUpdates.put("table_key", "table_value");

    UpdateMap tableUpdateMap = UpdateMap.builder().updates(tableUpdates).replace(false).build();

    UpdateConfig updateConfig =
        UpdateConfig.builder()
            .configUpdates(configUpdateMap)
            .tableMetadataUpdates(tableUpdateMap)
            .build();

    // Verify new API
    assertNotNull(updateConfig.configUpdates());
    assertNotNull(updateConfig.tableMetadataUpdates());
    assertNull(updateConfig.schemaMetadataUpdates());
    assertTrue(updateConfig.fieldMetadataUpdates().isEmpty());

    assertEquals("config_value", updateConfig.configUpdates().updates().get("config_key"));
    assertNull(updateConfig.configUpdates().updates().get("delete_key"));
    assertEquals("table_value", updateConfig.tableMetadataUpdates().updates().get("table_key"));
  }
}
