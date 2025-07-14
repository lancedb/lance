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
import com.lancedb.lance.operation.Project;

import org.apache.arrow.util.Preconditions;
import org.apache.arrow.vector.types.pojo.Schema;

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
  // Mainly for JNI usage
  private final Dataset dataset;
  private Operation operation;
  private Operation blobOp;

  private Transaction(
      Dataset dataset, long readVersion, String uuid, Map<String, String> writeParams) {
    this.dataset = dataset;
    this.readVersion = readVersion;
    this.uuid = uuid;
    this.writeParams = writeParams != null ? writeParams : new HashMap<>();
  }

  public Dataset dataset() {
    return dataset;
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

  public Transaction newProject(Schema schema) {
    validateState();
    this.operation = new Project.Builder().schema(schema).allocator(dataset.allocator).build();
    return this;
  }

  private void validateState() {
    Preconditions.checkState(operation == null, "Multiple operations are not supported yet");
  }

  public Dataset commit() {
    Preconditions.checkState(operation != null, "Transaction has no operations");
    try {
      Dataset committed = commitNative();
      committed.allocator = dataset.allocator;
      return committed;
    } finally {
      operation.release();
      if (blobOp != null) {
        blobOp.release();
      }
    }
  }

  private native Dataset commitNative();

  public static class Builder {
    private final String uuid;
    private final Dataset dataset;
    private long readVersion;
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

    public Transaction build() {
      return new Transaction(dataset, readVersion, uuid, writeParams);
    }
  }
}
