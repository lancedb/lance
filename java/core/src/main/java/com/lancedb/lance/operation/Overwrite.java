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

import com.lancedb.lance.FragmentMetadata;

import org.apache.arrow.vector.types.pojo.Schema;

import java.util.List;
import java.util.Map;

/**
 * Overwrite the dataset with new fragments. This operation will overwrite the existing dataset.
 * Note: 1. The operation won't delete table config keys which do not exist in configUpsertValues.
 * 2. If we want to create a new Dataset, use {@link com.lancedb.lance.Dataset}.create instead.
 */
public class Overwrite extends SchemaOperation {

  private final List<FragmentMetadata> fragments;
  private final Map<String, String> configUpsertValues;

  protected Overwrite(
      List<FragmentMetadata> fragments, Schema schema, Map<String, String> configUpsertValues) {
    super(schema);
    this.fragments = fragments;
    this.configUpsertValues = configUpsertValues;
  }

  public List<FragmentMetadata> fragments() {
    return fragments;
  }

  public Map<String, String> configUpsertValues() {
    return configUpsertValues;
  }

  public static Builder builder() {
    return new Builder();
  }

  @Override
  public String name() {
    return "Overwrite";
  }

  @Override
  public String toString() {
    return "Overwrite{"
        + "fragments="
        + fragments
        + ", schema="
        + schema()
        + ", configUpsertValues="
        + configUpsertValues
        + '}';
  }

  // Builder class for Overwrite
  public static class Builder {
    private List<FragmentMetadata> fragments;
    private Schema schema;
    private Map<String, String> configUpsertValues;

    private Builder() {}

    public Builder fragments(List<FragmentMetadata> fragments) {
      this.fragments = fragments;
      return this;
    }

    public Builder schema(Schema schema) {
      this.schema = schema;
      return this;
    }

    public Builder configUpsertValues(Map<String, String> configUpsertValues) {
      this.configUpsertValues = configUpsertValues;
      return this;
    }

    public Overwrite build() {
      return new Overwrite(fragments, schema, configUpsertValues);
    }
  }
}
