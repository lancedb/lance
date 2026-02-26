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

import org.lance.operation.Operation;

import org.apache.arrow.util.Preconditions;

import java.util.Map;
import java.util.Optional;

/**
 * A convenience wrapper that pairs a {@link Transaction} with a {@link Dataset}, providing a simple
 * commit workflow.
 *
 * <p>This replaces the old {@code Transaction} class's "sourced" role where the transaction held a
 * reference to the dataset it was built from.
 *
 * <p>Example usage:
 *
 * <pre>{@code
 * try (SourcedTransaction txn = dataset.newTransactionBuilder()
 *     .operation(Append.builder().fragments(fragments).build())
 *     .build();
 *     Dataset committed = txn.commit()) {
 *     // use committed dataset
 * }
 * }</pre>
 */
public class SourcedTransaction implements AutoCloseable {

  private final Transaction transaction;
  private final Dataset dataset;

  private SourcedTransaction(Transaction transaction, Dataset dataset) {
    this.transaction = transaction;
    this.dataset = dataset;
  }

  /** Returns the underlying {@link Transaction}. */
  public Transaction transaction() {
    return transaction;
  }

  /** Delegates to {@link Transaction#readVersion()}. */
  public long readVersion() {
    return transaction.readVersion();
  }

  /** Delegates to {@link Transaction#uuid()}. */
  public String uuid() {
    return transaction.uuid();
  }

  /** Delegates to {@link Transaction#operation()}. */
  public Operation operation() {
    return transaction.operation();
  }

  /** Delegates to {@link Transaction#tag()}. */
  public Optional<String> tag() {
    return transaction.tag();
  }

  /** Delegates to {@link Transaction#transactionProperties()}. */
  public Optional<Map<String, String>> transactionProperties() {
    return transaction.transactionProperties();
  }

  /**
   * Commit this transaction against the source dataset.
   *
   * @return a new Dataset at the committed version
   */
  public Dataset commit() {
    return dataset.commitTransaction(transaction);
  }

  /**
   * Commit this transaction against the source dataset with additional options.
   *
   * @param detached if true, the commit will not be part of the main dataset lineage
   * @param enableV2ManifestPaths if true, and this is a new dataset, uses the new V2 manifest paths
   * @return a new Dataset at the committed version
   */
  public Dataset commit(boolean detached, boolean enableV2ManifestPaths) {
    return dataset.commitTransaction(transaction, detached, enableV2ManifestPaths);
  }

  /** Release native resources held by the underlying transaction's operation. */
  @Override
  public void close() {
    transaction.close();
  }

  /** Builder for constructing {@link SourcedTransaction} instances from a {@link Dataset}. */
  public static class Builder {
    private final Dataset dataset;
    private long readVersion;
    private Operation operation;
    private String tag;
    private Map<String, String> transactionProperties;

    /**
     * Create a builder for committing against an existing dataset. The read version defaults to the
     * dataset's current version.
     *
     * @param dataset the existing dataset to commit against
     */
    public Builder(Dataset dataset) {
      this.dataset = dataset;
      this.readVersion = dataset.version();
    }

    public Builder readVersion(long readVersion) {
      this.readVersion = readVersion;
      return this;
    }

    public Builder operation(Operation operation) {
      if (this.operation != null) {
        throw new IllegalStateException(
            String.format("Operation %s has been set", this.operation.name()));
      }
      this.operation = operation;
      return this;
    }

    /**
     * Set an optional tag for the transaction.
     *
     * @param tag the tag string
     * @return this builder instance
     */
    public Builder tag(String tag) {
      this.tag = tag;
      return this;
    }

    public Builder transactionProperties(Map<String, String> properties) {
      this.transactionProperties = properties;
      return this;
    }

    public SourcedTransaction build() {
      Preconditions.checkState(operation != null, "TransactionBuilder has no operations");
      Transaction transaction =
          new Transaction.Builder()
              .readVersion(readVersion)
              .operation(operation)
              .tag(tag)
              .transactionProperties(transactionProperties)
              .build();
      return new SourcedTransaction(transaction, dataset);
    }
  }
}
