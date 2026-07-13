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

import java.util.Arrays;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;

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
  private final Optional<byte[]> rowAddressLayoutDelta;

  /**
   * Constructor used by JNI when reading transactions from native code.
   *
   * @param readVersion the version that was read when creating this transaction
   * @param uuid the unique identifier for this transaction
   * @param operation the operation to perform
   * @param tag optional tag for the transaction
   * @param transactionProperties optional transaction properties
   * @param rowAddressLayoutDelta opaque core-authored row-address provenance
   */
  private Transaction(
      long readVersion,
      String uuid,
      Operation operation,
      String tag,
      Map<String, String> transactionProperties,
      byte[] rowAddressLayoutDelta) {
    this.readVersion = readVersion;
    this.uuid = uuid;
    this.operation = operation;
    this.tag = Optional.ofNullable(tag);
    this.transactionProperties = Optional.ofNullable(transactionProperties);
    this.rowAddressLayoutDelta =
        Optional.ofNullable(rowAddressLayoutDelta).map(value -> value.clone());
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

  /**
   * Returns opaque core-authored row-address provenance for distributed format 2.3 writes.
   *
   * <p>The bytes must be preserved unchanged when a transaction is transported between workers.
   */
  public Optional<byte[]> rowAddressLayoutDelta() {
    return rowAddressLayoutDelta.map(value -> value.clone());
  }

  /** Release native resources held by the operation (e.g. Arrow C schemas). */
  @Override
  public void close() {
    operation.release();
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
        && Objects.equals(transactionProperties, that.transactionProperties)
        && Arrays.equals(
            rowAddressLayoutDelta.orElse(null), that.rowAddressLayoutDelta.orElse(null));
  }

  @Override
  public int hashCode() {
    return 31 * Objects.hash(readVersion, uuid, operation, tag, transactionProperties)
        + Arrays.hashCode(rowAddressLayoutDelta.orElse(null));
  }

  /** Builder for constructing {@link Transaction} instances. */
  public static class Builder {
    private String uuid;
    private long readVersion;
    private Operation operation;
    private String tag;
    private Map<String, String> transactionProperties;
    private byte[] rowAddressLayoutDelta;

    public Builder() {
      this.uuid = UUID.randomUUID().toString();
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

    /** Preserve opaque core-authored row-address provenance from another transaction. */
    public Builder rowAddressLayoutDelta(byte[] rowAddressLayoutDelta) {
      this.rowAddressLayoutDelta =
          rowAddressLayoutDelta == null ? null : rowAddressLayoutDelta.clone();
      return this;
    }

    public Transaction build() {
      Preconditions.checkState(operation != null, "TransactionBuilder has no operations");
      return new Transaction(
          readVersion, uuid, operation, tag, transactionProperties, rowAddressLayoutDelta);
    }
  }
}
