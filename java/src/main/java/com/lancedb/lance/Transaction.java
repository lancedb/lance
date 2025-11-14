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

import com.lancedb.lance.operation.Operation;

import com.google.common.base.MoreObjects;
import org.apache.arrow.util.Preconditions;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;

/**
 * Align with the Transaction struct in rust. The transaction won't commit the status to original
 * dataset. It will return a new dataset after committed.
 */
public class Transaction {

  private final long readVersion;
  private final String uuid;
  private final Map<String, String> writeParams;
  private final Optional<Map<String, String>> transactionProperties;
  // Mainly for JNI usage
  private final Dataset dataset;
  private final Operation operation;

  private Transaction(
      Dataset dataset,
      long readVersion,
      String uuid,
      Operation operation,
      Map<String, String> writeParams,
      Map<String, String> transactionProperties) {
    this.dataset = dataset;
    this.readVersion = readVersion;
    this.uuid = uuid;
    this.operation = operation;
    this.writeParams = writeParams != null ? writeParams : new HashMap<>();
    this.transactionProperties = Optional.ofNullable(transactionProperties);
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

  public Map<String, String> writeParams() {
    return writeParams;
  }

  public Optional<Map<String, String>> transactionProperties() {
    return transactionProperties;
  }

  public Dataset commit() {
    if (dataset == null) {
      throw new UnsupportedOperationException("Transaction doesn't support create new dataset yet");
    }
    return dataset.commitTransaction(this);
  }

  public void release() {
    operation.release();
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("readVersion", readVersion)
        .add("uuid", uuid)
        .add("operation", operation)
        .add("writeParams", writeParams)
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
        && Objects.equals(writeParams, that.writeParams)
        && Objects.equals(transactionProperties, that.transactionProperties);
  }

  public static class Builder {
    private final String uuid;
    private final Dataset dataset;
    private long readVersion;
    private Operation operation;
    private Map<String, String> writeParams;
    private Map<String, String> transactionProperties;

    public Builder(Dataset dataset) {
      this.dataset = dataset;
      this.uuid = UUID.randomUUID().toString();
    }

    public Builder readVersion(long readVersion) {
      this.readVersion = readVersion;
      return this;
    }

    public Builder transactionProperties(Map<String, String> properties) {
      this.transactionProperties = properties;
      return this;
    }

    public Builder writeParams(Map<String, String> writeParams) {
      this.writeParams = writeParams;
      return this;
    }

    public Builder operation(Operation operation) {
      validateState();
      this.operation = operation;
      return this;
    }

    private void validateState() {
      if (operation != null) {
        throw new IllegalStateException(
            String.format("Operation %s has been set", operation.name()));
      }
    }

    public Transaction build() {
      Preconditions.checkState(operation != null, "TransactionBuilder has no operations");
      return new Transaction(
          dataset, readVersion, uuid, operation, writeParams, transactionProperties);
    }
  }
}
