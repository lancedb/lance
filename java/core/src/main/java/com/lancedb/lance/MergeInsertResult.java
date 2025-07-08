/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License a
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

import java.util.Objects;

/**
 * Result of a merge insert operation.
 *
 * This class contains statistics about the merge insert operation,
 * including the number of rows inserted, updated, and deleted.
 */
public class MergeInsertResult {

    private final long numInsertedRows;
    private final long numUpdatedRows;
    private final long numDeletedRows;

    /**
     * Create a new MergeInsertResult.
     *
     * @param numInsertedRows the number of rows inserted
     * @param numUpdatedRows the number of rows updated
     * @param numDeletedRows the number of rows deleted
     */
    public MergeInsertResult(long numInsertedRows, long numUpdatedRows, long numDeletedRows) {
        this.numInsertedRows = numInsertedRows;
        this.numUpdatedRows = numUpdatedRows;
        this.numDeletedRows = numDeletedRows;
    }

    /**
     * Get the number of rows that were inserted.
     *
     * @return the number of inserted rows
     */
    public long getNumInsertedRows() {
        return numInsertedRows;
    }

    /**
     * Get the number of rows that were updated.
     *
     * @return the number of updated rows
     */
    public long getNumUpdatedRows() {
        return numUpdatedRows;
    }

    /**
     * Get the number of rows that were deleted.
     *
     * @return the number of deleted rows
     */
    public long getNumDeletedRows() {
        return numDeletedRows;
    }

    /**
     * Get the total number of rows affected by the operation.
     *
     * @return the total number of affected rows
     */
    public long getTotalAffectedRows() {
        return numInsertedRows + numUpdatedRows + numDeletedRows;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null || getClass() != obj.getClass()) {
            return false;
        }
        MergeInsertResult that = (MergeInsertResult) obj;
        return numInsertedRows == that.numInsertedRows &&
               numUpdatedRows == that.numUpdatedRows &&
               numDeletedRows == that.numDeletedRows;
    }

    @Override
    public int hashCode() {
        return Objects.hash(numInsertedRows, numUpdatedRows, numDeletedRows);
    }

    @Override
    public String toString() {
        return String.format("MergeInsertResult{inserted=%d, updated=%d, deleted=%d}",
                           numInsertedRows, numUpdatedRows, numDeletedRows);
    }
}