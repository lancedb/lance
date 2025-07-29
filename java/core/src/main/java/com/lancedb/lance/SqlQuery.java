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
  private String table;
  private boolean withRowId = false;
  private boolean withRowAddr = false;

  public SqlQuery(Dataset dataset, String sql) {
    this.dataset = dataset;
    this.sql = sql;
  }

  public SqlQuery tableName(String tableName) {
    this.table = tableName;
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
    try (ArrowArrayStream s = ArrowArrayStream.allocateNew(dataset.allocator)) {
      intoBatchRecords(
          dataset, sql, Optional.ofNullable(table), withRowId, withRowAddr, s.memoryAddress());
      return Data.importArrayStream(dataset.allocator, s);
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

  public String intoExplainPlan(boolean verbose, boolean analyze) throws IOException {
    return intoExplainPlan(
        dataset, sql, Optional.ofNullable(table), withRowId, withRowAddr, verbose, analyze);
  }

  private static native String intoExplainPlan(
      Dataset dataset,
      String sql,
      Optional<String> tableName,
      boolean withRowId,
      boolean withRowAddr,
      boolean verbose,
      boolean analyze)
      throws IOException;
}
