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
package com.lancedb.lance.schema;

import com.google.common.base.MoreObjects;
import org.apache.arrow.vector.types.pojo.Schema;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class LanceSchema {
  private final List<LanceField> fields;
  private final Map<String, String> metadata;

  LanceSchema(List<LanceField> fields, Map<String, String> metadata) {
    this.fields = fields;
    this.metadata = metadata;
  }

  public List<LanceField> fields() {
    return fields;
  }

  public Map<String, String> metadata() {
    return Collections.unmodifiableMap(metadata);
  }

  public Schema asArrowSchema() {
    return new Schema(
        fields.stream().map(LanceField::asArrowField).collect(Collectors.toList()), metadata);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("fields", fields)
        .add("metadata", metadata)
        .toString();
  }

  // Builder class for LanceSchema
  private static class Builder {
    private List<LanceField> fields;
    private Map<String, String> metadata;

    Builder() {}

    public Builder withFields(List<LanceField> fields) {
      this.fields = fields;
      return this;
    }

    public Builder withMetadata(Map<String, String> metadata) {
      this.metadata = metadata;
      return this;
    }

    public LanceSchema build() {
      return new LanceSchema(fields, metadata);
    }
  }
}
