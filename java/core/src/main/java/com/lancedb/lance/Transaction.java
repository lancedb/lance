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

import com.lancedb.lance.operation.Append;
import com.lancedb.lance.operation.Operation;
import com.lancedb.lance.operation.Overwrite;
import com.lancedb.lance.operation.Project;

import org.apache.arrow.util.Preconditions;
import org.apache.arrow.vector.types.pojo.Schema;

import java.util.HashMap;
import java.util.List;
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

  public Dataset commit() {
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
    private Operation.Builder<?> operationBuilder;
    private Operation.Builder<?> blobOpBuilder;
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

    public Builder operationBuilder(Operation.Builder<?> opBuilder) {
      validateState();
      this.operationBuilder = opBuilder;
      return this;
    }

    public Builder project(Schema newSchema) {
      validateState();
      this.operationBuilder = new Project.Builder(dataset.allocator).schema(newSchema);
      return this;
    }

    public Builder overwrite(List<FragmentMetadata> fragments, Schema schema) {
      validateState();
      this.operationBuilder =
          new Overwrite.Builder(dataset.allocator).fragments(fragments).schema(schema);
      return this;
    }

    /**
     * upsertTableConfig would be reused for both Overwrite and UpdateConfig operations.
     *
     * @param upsertTableConfig the table config want to be upsert
     * @return the builder
     */
    public Builder upsertTableConfig(Map<String, String> upsertTableConfig) {
      if (operationBuilder instanceof Overwrite.Builder) {
        operationBuilder = ((Overwrite.Builder) operationBuilder).configUpsertValues(upsertTableConfig);
      }
      // TODO: Reuse this for UpdateConfig operation
      return this;
    }

    public Builder append(List<FragmentMetadata> fragments) {
      validateState();
      this.operationBuilder = new Append.Builder().fragments(fragments);
      return this;
    }

    private void validateState() {
      if (operationBuilder != null) {
        throw new IllegalStateException(
            String.format("Operation %s has been set", operationBuilder.desc()));
      }
    }

    public Transaction build() {
      Preconditions.checkState(operationBuilder != null, "TransactionBuilder has no operations");
      return new Transaction(
          dataset,
          readVersion,
          uuid,
          operationBuilder.build(),
          blobOpBuilder != null ? blobOpBuilder.build() : null,
          writeParams);
    }
  }
}
