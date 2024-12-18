/*
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

package com.lancedb.lance.schema;

import org.apache.arrow.vector.types.pojo.ArrowType;

import java.io.Serializable;
import java.util.Optional;

/** Column alteration used to alter dataset columns. */
public class ColumnAlteration implements Serializable {

  private String path;
  private Optional<String> rename;
  private Optional<Boolean> nullable;
  private Optional<ArrowType> dataType;

  public ColumnAlteration() {
    this("");
  }

  public ColumnAlteration(String path) {
    this.path = path;
    this.rename = Optional.empty();
    this.nullable = Optional.empty();
    this.dataType = Optional.empty();
  }

  public String getPath() {
    return path;
  }

  public Optional<String> getRename() {
    return rename;
  }

  public Optional<Boolean> getNullable() {
    return nullable;
  }

  public Optional<ArrowType> getDataType() {
    return dataType;
  }

  public void setPath(String path) {
    this.path = path;
  }

  public void setRename(Optional<String> rename) {
    this.rename = rename;
  }

  public void setNullable(Optional<Boolean> nullable) {
    this.nullable = nullable;
  }

  public void setDataType(Optional<ArrowType> dataType) {
    this.dataType = dataType;
  }
}
