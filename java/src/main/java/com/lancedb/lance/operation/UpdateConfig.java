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

import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Update the dataset configuration. This operation allows updating configuration values, deleting
 * configuration keys, and modifying schema and field metadata.
 */
public class UpdateConfig implements Operation {
  private final Optional<Map<String, String>> upsertValues;
  private final Optional<List<String>> deleteKeys;
  private final Optional<Map<String, String>> schemaMetadata;
  private final Optional<Map<Integer, Map<String, String>>> fieldMetadata;

  private UpdateConfig(
      Map<String, String> upsertValues,
      List<String> deleteKeys,
      Map<String, String> schemaMetadata,
      Map<Integer, Map<String, String>> fieldMetadata) {
    this.upsertValues = Optional.ofNullable(upsertValues);
    this.deleteKeys = Optional.ofNullable(deleteKeys);
    this.schemaMetadata = Optional.ofNullable(schemaMetadata);
    this.fieldMetadata = Optional.ofNullable(fieldMetadata);
  }

  public Optional<Map<String, String>> upsertValues() {
    return upsertValues;
  }

  public Optional<List<String>> deleteKeys() {
    return deleteKeys;
  }

  public Optional<Map<String, String>> schemaMetadata() {
    return schemaMetadata;
  }

  public Optional<Map<Integer, Map<String, String>>> fieldMetadata() {
    return fieldMetadata;
  }

  @Override
  public String name() {
    return "UpdateConfig";
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("upsertValues", upsertValues)
        .add("deleteKeys", deleteKeys)
        .add("schemaMetadata", schemaMetadata)
        .add("fieldMetadata", fieldMetadata)
        .toString();
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder {
    private Map<String, String> upsertValues;
    private List<String> deleteKeys;
    private Map<String, String> schemaMetadata;
    private Map<Integer, Map<String, String>> fieldMetadata;

    public Builder() {}

    public Builder upsertValues(Map<String, String> upsertValues) {
      this.upsertValues = upsertValues;
      return this;
    }

    public Builder deleteKeys(List<String> deleteKeys) {
      this.deleteKeys = deleteKeys;
      return this;
    }

    public Builder schemaMetadata(Map<String, String> schemaMetadata) {
      this.schemaMetadata = schemaMetadata;
      return this;
    }

    public Builder fieldMetadata(Map<Integer, Map<String, String>> fieldMetadata) {
      this.fieldMetadata = fieldMetadata;
      return this;
    }

    public UpdateConfig build() {
      return new UpdateConfig(upsertValues, deleteKeys, schemaMetadata, fieldMetadata);
    }
  }
}
