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
package org.lance;

import com.google.common.base.MoreObjects;

import java.util.Objects;

/**
 * Stable identity for a field-scoped Boolean Cell Flag.
 *
 * <pre>{@code
 * CellFlagDefinition definition =
 *     dataset.registerCellFlag("embedding", "computed", false);
 * long stableId = definition.flagId();
 * }</pre>
 */
public final class CellFlagDefinition {
  private final long flagId;
  private final int fieldId;
  private final String name;

  /** Constructor used by JNI. */
  private CellFlagDefinition(long flagId, int fieldId, String name) {
    this.flagId = flagId;
    this.fieldId = fieldId;
    this.name = name;
  }

  /** Returns the dataset-unique ID, which is never reused. */
  public long flagId() {
    return flagId;
  }

  /** Returns the stable Lance schema field ID. */
  public int fieldId() {
    return fieldId;
  }

  /** Returns the user-visible name, unique for the field. */
  public String name() {
    return name;
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof CellFlagDefinition)) {
      return false;
    }
    CellFlagDefinition that = (CellFlagDefinition) other;
    return flagId == that.flagId && fieldId == that.fieldId && name.equals(that.name);
  }

  @Override
  public int hashCode() {
    return Objects.hash(flagId, fieldId, name);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("flagId", flagId)
        .add("fieldId", fieldId)
        .add("name", name)
        .toString();
  }
}
