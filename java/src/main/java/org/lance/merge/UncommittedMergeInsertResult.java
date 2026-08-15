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
package org.lance.merge;

import org.lance.Dataset;
import org.lance.Transaction;

import com.google.common.base.MoreObjects;
import org.apache.arrow.util.Preconditions;

/**
 * Result of {@link org.lance.Dataset#mergeInsertUncommitted(MergeInsertParams,
 * org.apache.arrow.c.ArrowArrayStream)}.
 *
 * <p>Contains the dataset on which merge insert was performed, the uncommitted {@link Transaction},
 * and the execution {@link MergeInsertStats}.
 */
public class UncommittedMergeInsertResult implements AutoCloseable {
  private final Dataset dataset;
  private final Transaction transaction;
  private final MergeInsertStats stats;
  private final byte[] affectedRows;

  public UncommittedMergeInsertResult(
      Dataset dataset, Transaction transaction, MergeInsertStats stats, byte[] affectedRows) {
    this.dataset = Preconditions.checkNotNull(dataset);
    this.transaction = Preconditions.checkNotNull(transaction);
    this.stats = Preconditions.checkNotNull(stats);
    this.affectedRows = affectedRows;
  }

  public UncommittedMergeInsertResult(
      Dataset dataset, Transaction transaction, MergeInsertStats stats) {
    this(dataset, transaction, stats, null);
  }

  public Dataset dataset() {
    return dataset;
  }

  public Dataset getDataset() {
    return dataset;
  }

  public Transaction transaction() {
    return transaction;
  }

  public Transaction getTransaction() {
    return transaction;
  }

  public MergeInsertStats stats() {
    return stats;
  }

  public MergeInsertStats getStats() {
    return stats;
  }

  public byte[] affectedRows() {
    return affectedRows;
  }

  public byte[] getAffectedRows() {
    return affectedRows;
  }

  @Override
  public void close() {
    transaction.close();
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("dataset", dataset)
        .add("transaction", transaction)
        .add("stats", stats)
        .add("hasAffectedRows", affectedRows != null)
        .toString();
  }
}
