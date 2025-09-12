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
package com.lancedb.lance.merge;

import com.google.common.base.MoreObjects;

public final class MergeInsertStats {
  private final long numInsertedRows;
  private final long numUpdatedRows;
  private final long numDeletedRows;
  private final int numAttempts;
  private final long bytesWritten;
  private final long numFilesWritten;

  public MergeInsertStats(
      long numInsertedRows,
      long numUpdatedRows,
      long numDeletedRows,
      int numAttempts,
      long bytesWritten,
      long numFilesWritten) {
    this.numInsertedRows = numInsertedRows;
    this.numUpdatedRows = numUpdatedRows;
    this.numDeletedRows = numDeletedRows;
    this.numAttempts = numAttempts;
    this.bytesWritten = bytesWritten;
    this.numFilesWritten = numFilesWritten;
  }

  public long numInsertedRows() {
    return numInsertedRows;
  }

  public long numUpdatedRows() {
    return numUpdatedRows;
  }

  public long numDeletedRows() {
    return numDeletedRows;
  }

  public int numAttempts() {
    return numAttempts;
  }

  public long bytesWritten() {
    return bytesWritten;
  }

  public long numFilesWritten() {
    return numFilesWritten;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("numInsertedRows", numInsertedRows)
        .add("numUpdatedRows", numUpdatedRows)
        .add("numDeletedRows", numDeletedRows)
        .add("numAttempts", numAttempts)
        .add("bytesWritten", bytesWritten)
        .add("numFilesWritten", numFilesWritten)
        .toString();
  }
}
