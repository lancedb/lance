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
import com.google.common.base.Preconditions;

/**
 * An explicit Boolean change to a registered field-scoped Cell Flag.
 *
 * <pre>{@code
 * UpdateParams params = UpdateParams.forCellFlags(
 *     new CellFlagChange("embedding", "computed", true));
 * }</pre>
 */
public final class CellFlagChange {
  private final String field;
  private final String name;
  private final boolean value;

  public CellFlagChange(String field, String name, boolean value) {
    this.field = Preconditions.checkNotNull(field, "field must not be null");
    this.name = Preconditions.checkNotNull(name, "name must not be null");
    this.value = value;
  }

  public String field() {
    return field;
  }

  public String name() {
    return name;
  }

  public boolean value() {
    return value;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("field", field)
        .add("name", name)
        .add("value", value)
        .toString();
  }
}
