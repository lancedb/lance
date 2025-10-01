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

import com.google.common.base.MoreObjects;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Update the dataset configuration. This operation allows updating configuration values, schema
 * metadata, table metadata, and field metadata using the UpdateMap pattern.
 *
 * <h2>Migration Guide from Legacy API</h2>
 *
 * <p>This class has been updated to use the new {@link UpdateMap} pattern for all metadata updates.
 * The old API methods are deprecated but still supported for backward compatibility.
 *
 * <h3>Old API (Deprecated):</h3>
 *
 * <pre>{@code
 * UpdateConfig.builder()
 *     .upsertValues(Map.of("key1", "value1", "key2", "value2"))
 *     .deleteKeys(List.of("old_key1", "old_key2"))
 *     .schemaMetadata(Map.of("schema_key", "schema_value"))
 *     .fieldMetadata(Map.of(0, Map.of("field_key", "field_value")))
 *     .build();
 * }</pre>
 *
 * <h3>New API (Recommended):</h3>
 *
 * <pre>{@code
 * // For config updates (combines upsert and delete)
 * UpdateMap configUpdates = UpdateMap.builder()
 *     .updates(Map.of("key1", "value1", "key2", "value2", "old_key1", null, "old_key2", null))
 *     .replace(false)  // false = incremental updates, true = replace all
 *     .build();
 *
 * // For table metadata (new feature!)
 * UpdateMap tableMetadata = UpdateMap.builder()
 *     .updates(Map.of("dataset_description", "My dataset", "version", "1.0"))
 *     .replace(false)
 *     .build();
 *
 * // For schema metadata
 * UpdateMap schemaMetadata = UpdateMap.builder()
 *     .updates(Map.of("schema_key", "schema_value"))
 *     .replace(false)
 *     .build();
 *
 * // For field metadata
 * UpdateMap field0Metadata = UpdateMap.builder()
 *     .updates(Map.of("field_key", "field_value"))
 *     .replace(false)
 *     .build();
 *
 * UpdateConfig.builder()
 *     .configUpdates(configUpdates)
 *     .tableMetadataUpdates(tableMetadata)      // NEW: Table-level metadata
 *     .schemaMetadataUpdates(schemaMetadata)
 *     .fieldMetadataUpdates(Map.of(0, field0Metadata))
 *     .build();
 * }</pre>
 */
public class UpdateConfig implements Operation {
  private final UpdateMap configUpdates;
  private final UpdateMap tableMetadataUpdates;
  private final UpdateMap schemaMetadataUpdates;
  private final Map<Integer, UpdateMap> fieldMetadataUpdates;

  private UpdateConfig(
      UpdateMap configUpdates,
      UpdateMap tableMetadataUpdates,
      UpdateMap schemaMetadataUpdates,
      Map<Integer, UpdateMap> fieldMetadataUpdates) {
    this.configUpdates = configUpdates;
    this.tableMetadataUpdates = tableMetadataUpdates;
    this.schemaMetadataUpdates = schemaMetadataUpdates;
    this.fieldMetadataUpdates =
        fieldMetadataUpdates != null ? fieldMetadataUpdates : new HashMap<>();
  }

  /**
   * @deprecated Use the new constructor with UpdateMap parameters instead. This constructor
   *     converts old-style parameters to the new UpdateMap structure.
   * @param upsertValues Map of config key-value pairs to upsert
   * @param deleteKeys List of config keys to delete
   * @param schemaMetadata Map of schema metadata to update
   * @param fieldMetadata Map of field metadata to update
   */
  @Deprecated
  private UpdateConfig(
      Map<String, String> upsertValues,
      List<String> deleteKeys,
      Map<String, String> schemaMetadata,
      Map<Integer, Map<String, String>> fieldMetadata) {

    // Convert upsert values and delete keys to UpdateMap
    if (upsertValues != null || deleteKeys != null) {
      Map<String, String> configMap = new HashMap<>();
      if (upsertValues != null) {
        configMap.putAll(upsertValues);
      }
      if (deleteKeys != null) {
        for (String key : deleteKeys) {
          configMap.put(key, null);
        }
      }
      this.configUpdates =
          configMap.isEmpty()
              ? null
              : UpdateMap.builder().updates(configMap).replace(false).build();
    } else {
      this.configUpdates = null;
    }

    // Table metadata was not supported in the old API
    this.tableMetadataUpdates = null;

    // Convert schema metadata
    this.schemaMetadataUpdates =
        (schemaMetadata == null || schemaMetadata.isEmpty())
            ? null
            : UpdateMap.builder().updates(schemaMetadata).replace(false).build();

    // Convert field metadata
    if (fieldMetadata == null || fieldMetadata.isEmpty()) {
      this.fieldMetadataUpdates = new HashMap<>();
    } else {
      Map<Integer, UpdateMap> convertedFieldMetadata = new HashMap<>();
      for (Map.Entry<Integer, Map<String, String>> entry : fieldMetadata.entrySet()) {
        if (entry.getValue() != null && !entry.getValue().isEmpty()) {
          UpdateMap fieldUpdateMap =
              UpdateMap.builder().updates(entry.getValue()).replace(false).build();
          convertedFieldMetadata.put(entry.getKey(), fieldUpdateMap);
        }
      }
      this.fieldMetadataUpdates = convertedFieldMetadata;
    }
  }

  public UpdateMap configUpdates() {
    return configUpdates;
  }

  public UpdateMap tableMetadataUpdates() {
    return tableMetadataUpdates;
  }

  public UpdateMap schemaMetadataUpdates() {
    return schemaMetadataUpdates;
  }

  public Map<Integer, UpdateMap> fieldMetadataUpdates() {
    return fieldMetadataUpdates;
  }

  // ===============================================================
  // DEPRECATED BACKWARD-COMPATIBLE METHODS
  // ===============================================================

  /**
   * @deprecated Use {@link #configUpdates()} instead. This method extracts upsert values from the
   *     new UpdateMap structure.
   * @return Optional containing upsert values from config updates, or empty if none
   */
  @Deprecated
  public Optional<Map<String, String>> upsertValues() {
    if (configUpdates == null) {
      return Optional.empty();
    }
    Map<String, String> upsertMap = new HashMap<>();
    for (Map.Entry<String, String> entry : configUpdates.updates().entrySet()) {
      if (entry.getValue() != null) {
        upsertMap.put(entry.getKey(), entry.getValue());
      }
    }
    return upsertMap.isEmpty() ? Optional.empty() : Optional.of(upsertMap);
  }

  /**
   * @deprecated Use {@link #configUpdates()} instead. This method extracts delete keys from the new
   *     UpdateMap structure.
   * @return Optional containing delete keys from config updates, or empty if none
   */
  @Deprecated
  public Optional<List<String>> deleteKeys() {
    if (configUpdates == null) {
      return Optional.empty();
    }
    List<String> deleteList =
        configUpdates.updates().entrySet().stream()
            .filter(entry -> entry.getValue() == null)
            .map(Map.Entry::getKey)
            .collect(java.util.stream.Collectors.toList());
    return deleteList.isEmpty() ? Optional.empty() : Optional.of(deleteList);
  }

  /**
   * @deprecated Use {@link #schemaMetadataUpdates()} instead. This method extracts updates from the
   *     new UpdateMap structure.
   * @return Optional containing schema metadata updates, or empty if none
   */
  @Deprecated
  public Optional<Map<String, String>> schemaMetadata() {
    if (schemaMetadataUpdates == null) {
      return Optional.empty();
    }
    Map<String, String> metadataMap = new HashMap<>();
    for (Map.Entry<String, String> entry : schemaMetadataUpdates.updates().entrySet()) {
      if (entry.getValue() != null) {
        metadataMap.put(entry.getKey(), entry.getValue());
      }
    }
    return metadataMap.isEmpty() ? Optional.empty() : Optional.of(metadataMap);
  }

  /**
   * @deprecated Use {@link #fieldMetadataUpdates()} instead. This method converts from the new
   *     UpdateMap structure.
   * @return Optional containing field metadata updates, or empty if none
   */
  @Deprecated
  public Optional<Map<Integer, Map<String, String>>> fieldMetadata() {
    if (fieldMetadataUpdates == null || fieldMetadataUpdates.isEmpty()) {
      return Optional.empty();
    }
    Map<Integer, Map<String, String>> legacyFieldMetadata = new HashMap<>();
    for (Map.Entry<Integer, UpdateMap> entry : fieldMetadataUpdates.entrySet()) {
      Map<String, String> fieldMap = new HashMap<>();
      for (Map.Entry<String, String> updateEntry : entry.getValue().updates().entrySet()) {
        if (updateEntry.getValue() != null) {
          fieldMap.put(updateEntry.getKey(), updateEntry.getValue());
        }
      }
      if (!fieldMap.isEmpty()) {
        legacyFieldMetadata.put(entry.getKey(), fieldMap);
      }
    }
    return legacyFieldMetadata.isEmpty() ? Optional.empty() : Optional.of(legacyFieldMetadata);
  }

  @Override
  public String name() {
    return "UpdateConfig";
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("configUpdates", configUpdates)
        .add("tableMetadataUpdates", tableMetadataUpdates)
        .add("schemaMetadataUpdates", schemaMetadataUpdates)
        .add("fieldMetadataUpdates", fieldMetadataUpdates)
        .toString();
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder {
    private UpdateMap configUpdates;
    private UpdateMap tableMetadataUpdates;
    private UpdateMap schemaMetadataUpdates;
    private Map<Integer, UpdateMap> fieldMetadataUpdates;

    public Builder() {}

    public Builder configUpdates(UpdateMap configUpdates) {
      this.configUpdates = configUpdates;
      return this;
    }

    public Builder tableMetadataUpdates(UpdateMap tableMetadataUpdates) {
      this.tableMetadataUpdates = tableMetadataUpdates;
      return this;
    }

    public Builder schemaMetadataUpdates(UpdateMap schemaMetadataUpdates) {
      this.schemaMetadataUpdates = schemaMetadataUpdates;
      return this;
    }

    public Builder fieldMetadataUpdates(Map<Integer, UpdateMap> fieldMetadataUpdates) {
      this.fieldMetadataUpdates = fieldMetadataUpdates;
      return this;
    }

    public UpdateConfig build() {
      return new UpdateConfig(
          configUpdates, tableMetadataUpdates, schemaMetadataUpdates, fieldMetadataUpdates);
    }

    // ===============================================================
    // DEPRECATED BACKWARD-COMPATIBLE BUILDER METHODS
    // ===============================================================

    /**
     * @deprecated Use {@link #configUpdates(UpdateMap)} instead. This method converts the old API
     *     to the new UpdateMap structure.
     * @param upsertValues Map of key-value pairs to upsert in the config
     * @return this builder
     */
    @Deprecated
    public Builder upsertValues(Map<String, String> upsertValues) {
      if (upsertValues != null && !upsertValues.isEmpty()) {
        this.configUpdates = UpdateMap.builder().updates(upsertValues).replace(false).build();
      }
      return this;
    }

    /**
     * @deprecated Use {@link #configUpdates(UpdateMap)} instead. This method converts the old API
     *     to the new UpdateMap structure.
     * @param deleteKeys List of keys to delete from the config
     * @return this builder
     */
    @Deprecated
    public Builder deleteKeys(List<String> deleteKeys) {
      if (deleteKeys != null && !deleteKeys.isEmpty()) {
        // Create a map with null values to indicate deletion
        Map<String, String> deleteMap = new HashMap<>();
        for (String key : deleteKeys) {
          deleteMap.put(key, null);
        }

        // If we already have config updates, merge the deletes
        if (this.configUpdates != null) {
          Map<String, String> combinedUpdates = new HashMap<>(this.configUpdates.updates());
          combinedUpdates.putAll(deleteMap);
          this.configUpdates =
              UpdateMap.builder()
                  .updates(combinedUpdates)
                  .replace(this.configUpdates.replace())
                  .build();
        } else {
          this.configUpdates = UpdateMap.builder().updates(deleteMap).replace(false).build();
        }
      }
      return this;
    }

    /**
     * @deprecated Use {@link #schemaMetadataUpdates(UpdateMap)} instead. This method converts the
     *     old API to the new UpdateMap structure.
     * @param schemaMetadata Map of schema metadata to update
     * @return this builder
     */
    @Deprecated
    public Builder schemaMetadata(Map<String, String> schemaMetadata) {
      if (schemaMetadata != null && !schemaMetadata.isEmpty()) {
        this.schemaMetadataUpdates =
            UpdateMap.builder().updates(schemaMetadata).replace(false).build();
      }
      return this;
    }

    /**
     * @deprecated Use {@link #fieldMetadataUpdates(Map)} instead. This method converts the old API
     *     to the new UpdateMap structure.
     * @param fieldMetadata Map of field ID to metadata map
     * @return this builder
     */
    @Deprecated
    public Builder fieldMetadata(Map<Integer, Map<String, String>> fieldMetadata) {
      if (fieldMetadata != null && !fieldMetadata.isEmpty()) {
        Map<Integer, UpdateMap> convertedFieldMetadata = new HashMap<>();
        for (Map.Entry<Integer, Map<String, String>> entry : fieldMetadata.entrySet()) {
          if (entry.getValue() != null && !entry.getValue().isEmpty()) {
            UpdateMap fieldUpdateMap =
                UpdateMap.builder().updates(entry.getValue()).replace(false).build();
            convertedFieldMetadata.put(entry.getKey(), fieldUpdateMap);
          }
        }
        this.fieldMetadataUpdates = convertedFieldMetadata;
      }
      return this;
    }
  }
}
