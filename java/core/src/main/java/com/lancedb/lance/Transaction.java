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

import org.apache.arrow.util.Preconditions;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

/**
 * Align with the Transaction struct in rust. The transaction won't commit the status to original
 * dataset. It will return a new dataset after committed.
 */
public class Transaction {

  private final long readVersion;
  private final String uuid;
  private final Map<String, String> writeParams;
  private final Dataset dataset;
  private final Operation operation;
  private final Operation blobOp;

  private Transaction(
      Dataset dataset,
      long readVersion,
      String uuid,
      Operation operation,
      Operation blobOp,
      Map<String, String> writeParams) {
    this.dataset = dataset;
    this.readVersion = readVersion;
    this.uuid = uuid;
    this.operation = operation;
    this.blobOp = blobOp;
    this.writeParams = writeParams != null ? writeParams : new HashMap<>();
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

  public Operation blobsOperation() {
    return blobOp;
  }

  public Map<String, String> writeParams() {
    return writeParams;
  }

  public Dataset commit() {
    if (dataset == null) {
      throw new UnsupportedOperationException("Transaction doesn't support create new dataset yet");
    }
    return dataset.commitTransaction(this);
  }

  public void release() {
    operation.release();
    if (blobOp != null) {
      blobOp.release();
    }
  }

  public static class Builder {
    private final String uuid;
    private final Dataset dataset;
    private long readVersion;
    private Operation operation;
    private Operation blobOp;
    private Map<String, String> writeParams;

    public Builder(Dataset dataset) {
      this.dataset = dataset;
      this.uuid = UUID.randomUUID().toString();
    }

    public Builder readVersion(long readVersion) {
      this.readVersion = readVersion;
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
      return new Transaction(dataset, readVersion, uuid, operation, blobOp, writeParams);
    }
  }
}
