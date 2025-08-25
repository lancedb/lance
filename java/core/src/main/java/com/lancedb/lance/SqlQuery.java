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
package com.lancedb.lance;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.vector.ipc.ArrowReader;

import java.io.IOException;
import java.util.Optional;

public class SqlQuery {
  private Dataset dataset;
  private String sql;
  private Optional<String> table = Optional.empty();
  private boolean withRowId = false;
  private boolean withRowAddr = false;

  public SqlQuery(Dataset dataset, String sql) {
    this.dataset = dataset;
    this.sql = sql;
  }

  /**
   * Specify a "table name" for the dataset, so that you can run SQL queries against it. In most
   * cases, we should not directly set the table_name. Instead, use {{DATASET}} as a placeholder for
   * the table name.
   *
   * <p>Example
   *
   * <pre>
   * String sql = "SELECT * FROM {{DATASET}} WHERE age > 20";
   * </pre>
   *
   * If you must set a table name, try to use a name that is unlikely to conflict, otherwise we may
   * encounter a 'table already exists' error.
   */
  public SqlQuery tableName(String tableName) {
    this.table = Optional.ofNullable(tableName);
    return this;
  }

  public SqlQuery withRowId(boolean withRowId) {
    this.withRowId = withRowId;
    return this;
  }

  public SqlQuery withRowAddr(boolean withAddr) {
    this.withRowAddr = withAddr;
    return this;
  }

  public ArrowReader intoBatchRecords() throws IOException {
    try (ArrowArrayStream s = ArrowArrayStream.allocateNew(dataset.allocator())) {
      intoBatchRecords(dataset, sql, table, withRowId, withRowAddr, s.memoryAddress());
      return Data.importArrayStream(dataset.allocator(), s);
    }
  }

  private static native void intoBatchRecords(
      Dataset dataset,
      String sql,
      Optional<String> tableName,
      boolean withRowId,
      boolean withRowAddr,
      long streamAddress)
      throws IOException;
}
