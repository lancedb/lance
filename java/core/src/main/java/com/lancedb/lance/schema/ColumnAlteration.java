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

import com.lancedb.lance.util.ToStringHelper;

import org.apache.arrow.util.Preconditions;
import org.apache.arrow.vector.types.pojo.ArrowType;

import java.util.Optional;

/** Column alteration used to alter dataset columns. */
public class ColumnAlteration {

  private final String path;
  private final String rename;
  private final Boolean nullable;
  private final ArrowType dataType;

  private ColumnAlteration(String path, String rename, Boolean nullable, ArrowType dataType) {
    this.path = path;
    this.rename = rename;
    this.nullable = nullable;
    this.dataType = dataType;
  }

  public String getPath() {
    return path;
  }

  public Optional<String> getRename() {
    return Optional.ofNullable(rename);
  }

  public Optional<Boolean> getNullable() {
    return Optional.ofNullable(nullable);
  }

  public Optional<ArrowType> getDataType() {
    return Optional.ofNullable(dataType);
  }

  @Override
  public String toString() {
    return ToStringHelper.of(this)
        .add("path", path)
        .add("rename", rename)
        .add("nullable", nullable)
        .add("dataType", dataType)
        .toString();
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder {

    private String path;
    private String rename;
    private Boolean nullable;
    private ArrowType dataType;

    public Builder() {}

    public Builder(String path) {
      this.path = path;
    }

    public Builder rename(String rename) {
      this.rename = rename;
      return this;
    }

    public Builder nullable(boolean nullable) {
      this.nullable = nullable;
      return this;
    }

    public Builder castTo(ArrowType dataType) {
      this.dataType = dataType;
      return this;
    }

    public ColumnAlteration build() {
      Preconditions.checkArgument(path != null, "path is required");
      return new ColumnAlteration(path, rename, nullable, dataType);
    }
  }
}
