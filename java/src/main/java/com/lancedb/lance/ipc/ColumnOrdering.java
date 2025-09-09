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
package com.lancedb.lance.ipc;

import org.apache.arrow.util.Preconditions;

import java.io.Serializable;

public class ColumnOrdering implements Serializable {
  private static final long serialVersionUID = 1L;
  private final String columnName;
  private final boolean nullFirst;
  private final boolean ascending;

  private ColumnOrdering(Builder builder) {
    this.columnName = Preconditions.checkNotNull(builder.columnName, "Columns must be set");
    Preconditions.checkArgument(!builder.columnName.isEmpty(), "Column must not be empty");
    this.nullFirst = builder.nullFirst;
    this.ascending = builder.ascending;
  }

  public String getColumnName() {
    return columnName;
  }

  public boolean isNullFirst() {
    return nullFirst;
  }

  public boolean isAscending() {
    return ascending;
  }

  @Override
  public String toString() {
    return "ColumnOrdering{"
        + "columnName='"
        + columnName
        + '\''
        + ", nullFirst="
        + nullFirst
        + ", ascending="
        + ascending
        + '}';
  }

  public static class Builder {
    private String columnName;
    private boolean nullFirst = true;
    private boolean ascending = true;

    public void setColumnName(String columnName) {
      this.columnName = columnName;
    }

    public void setNullFirst(boolean nullFirst) {
      this.nullFirst = nullFirst;
    }

    public void setAscending(boolean ascending) {
      this.ascending = ascending;
    }

    public ColumnOrdering build() {
      return new ColumnOrdering(this);
    }
  }
}
