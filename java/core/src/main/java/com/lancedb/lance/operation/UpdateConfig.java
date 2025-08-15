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

import java.util.Map;

/**
 * Update the dataset configuration. This operation allows updating configuration values, schema
 * metadata, table metadata, and field metadata using the UpdateMap pattern.
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
    this.fieldMetadataUpdates = fieldMetadataUpdates;
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

  @Override
  public String name() {
    return "UpdateConfig";
  }

  @Override
  public String toString() {
    return "UpdateConfig{"
        + "configUpdates="
        + configUpdates
        + ", tableMetadataUpdates="
        + tableMetadataUpdates
        + ", schemaMetadataUpdates="
        + schemaMetadataUpdates
        + ", fieldMetadataUpdates="
        + fieldMetadataUpdates
        + '}';
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
  }
}
