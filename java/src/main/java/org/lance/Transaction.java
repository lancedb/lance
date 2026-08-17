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

import com.google.common.base.MoreObjects;
import org.apache.arrow.util.Preconditions;

import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A pure data container representing a Lance transaction.
 *
 * <p>A Transaction holds the read version, a unique identifier, the operation to perform, and
 * optional transaction properties. It does not contain commit configuration or execution logic.
 *
 * <p>To commit a transaction, use {@link CommitBuilder} or {@link SourcedTransaction}.
 */
public class Transaction implements AutoCloseable {

  private final long readVersion;
  private final String uuid;
  private final Operation operation;
  private final Optional<String> tag;
  private final Optional<Map<String, String>> transactionProperties;
  private final String cellFlagTransactionPayload;
  private final OperationLease operationLease;
  private final AtomicBoolean closed = new AtomicBoolean();

  private static final class OperationLease {
    private final Operation operation;
    private final AtomicInteger references = new AtomicInteger(1);

    private OperationLease(Operation operation) {
      this.operation = operation;
    }

    private OperationLease retain() {
      int current = references.get();
      while (current > 0) {
        if (references.compareAndSet(current, current + 1)) {
          return this;
        }
        current = references.get();
      }
      throw new IllegalStateException("Transaction operation has already been released");
    }

    private void release() {
      if (references.decrementAndGet() == 0) {
        operation.release();
      }
    }
  }

  /**
   * Constructor used by JNI when reading transactions from native code.
   *
   * @param readVersion the version that was read when creating this transaction
   * @param uuid the unique identifier for this transaction
   * @param operation the operation to perform
   * @param tag optional tag for the transaction
   * @param transactionProperties optional transaction properties
   * @param cellFlagTransactionPayload opaque internal payload reserved for JNI
   */
  private Transaction(
      long readVersion,
      String uuid,
      Operation operation,
      String tag,
      Map<String, String> transactionProperties,
      String cellFlagTransactionPayload) {
    this(
        readVersion,
        uuid,
        operation,
        tag,
        transactionProperties,
        cellFlagTransactionPayload,
        new OperationLease(operation));
  }

  private Transaction(
      long readVersion,
      String uuid,
      Operation operation,
      String tag,
      Map<String, String> transactionProperties,
      String cellFlagTransactionPayload,
      OperationLease operationLease) {
    this.readVersion = readVersion;
    this.uuid = uuid;
    this.operation = operation;
    this.tag = Optional.ofNullable(tag);
    this.transactionProperties = Optional.ofNullable(transactionProperties);
    this.cellFlagTransactionPayload = cellFlagTransactionPayload;
    this.operationLease = operationLease;
  }

  /**
   * Create a transaction with the given read version and operation. A random UUID is generated
   * automatically.
   *
   * @param readVersion the version that was read when creating this transaction
   * @param operation the operation to perform
   */
  public Transaction(long readVersion, Operation operation) {
    this(readVersion, UUID.randomUUID().toString(), operation, null, null, null);
  }

  public long readVersion() {
    return readVersion;
  }

  public String uuid() {
    return uuid;
  }

  public Operation operation() {
    return operation;
  }

  /** Returns the optional tag for this transaction. */
  public Optional<String> tag() {
    return tag;
  }

  public Optional<Map<String, String>> transactionProperties() {
    return transactionProperties;
  }

  /** Release native resources held by the operation (e.g. Arrow C schemas). */
  @Override
  public synchronized void close() {
    if (closed.compareAndSet(false, true)) {
      operationLease.release();
    }
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("readVersion", readVersion)
        .add("uuid", uuid)
        .add("operation", operation)
        .add("tag", tag)
        .add("transactionProperties", transactionProperties)
        .toString();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    Transaction that = (Transaction) o;
    return readVersion == that.readVersion
        && uuid.equals(that.uuid)
        && Objects.equals(operation, that.operation)
        && Objects.equals(tag, that.tag)
        && Objects.equals(transactionProperties, that.transactionProperties);
  }

  @Override
  public int hashCode() {
    return Objects.hash(readVersion, uuid, operation, tag, transactionProperties);
  }

  /** Builder for constructing {@link Transaction} instances. */
  public static class Builder {
    private String uuid;
    private long readVersion;
    private Operation operation;
    private boolean inheritedOperation;
    private String tag;
    private Map<String, String> transactionProperties;
    private String cellFlagTransactionPayload;
    private OperationLease operationLease;
    private boolean built;

    public Builder() {
      this.uuid = UUID.randomUUID().toString();
    }

    /**
     * Create a builder initialized from an existing transaction.
     *
     * <p>Opaque internal metadata is preserved automatically. It cannot be read or supplied by
     * applications, which prevents an otherwise valid Cell Flag transaction from becoming a public
     * no-op when its tag, properties, read version, or operation is edited.
     */
    public Builder(Transaction transaction) {
      Preconditions.checkNotNull(transaction, "transaction must not be null");
      this.uuid = transaction.uuid;
      this.readVersion = transaction.readVersion;
      this.operation = transaction.operation;
      this.inheritedOperation = true;
      this.tag = transaction.tag.orElse(null);
      this.transactionProperties = transaction.transactionProperties.orElse(null);
      this.cellFlagTransactionPayload = transaction.cellFlagTransactionPayload;
      this.operationLease = transaction.operationLease;
    }

    public Builder readVersion(long readVersion) {
      this.readVersion = readVersion;
      return this;
    }

    public Builder uuid(String uuid) {
      this.uuid = uuid;
      return this;
    }

    public Builder operation(Operation operation) {
      if (this.operation != null && !inheritedOperation) {
        throw new IllegalStateException(
            String.format("Operation %s has been set", this.operation.name()));
      }
      if (inheritedOperation) {
        operationLease = null;
      }
      this.operation = operation;
      this.inheritedOperation = false;
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

    public Transaction build() {
      Preconditions.checkState(!built, "TransactionBuilder has already built a transaction");
      Preconditions.checkState(operation != null, "TransactionBuilder has no operations");
      built = true;
      OperationLease lease =
          inheritedOperation ? operationLease.retain() : new OperationLease(operation);
      return new Transaction(
          readVersion,
          uuid,
          operation,
          tag,
          transactionProperties,
          cellFlagTransactionPayload,
          lease);
    }
  }
}
