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

package com.lancedb.lance.spark.source;

import com.lancedb.lance.Dataset;
import com.lancedb.lance.Fragment;
import com.lancedb.lance.FragmentMetadata;
import com.lancedb.lance.FragmentOperation;
import com.lancedb.lance.WriteParams;
import com.lancedb.lance.spark.SparkCatalog;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.write.BatchWrite;
import org.apache.spark.sql.connector.write.DataWriter;
import org.apache.spark.sql.connector.write.DataWriterFactory;
import org.apache.spark.sql.connector.write.LogicalWriteInfo;
import org.apache.spark.sql.connector.write.PhysicalWriteInfo;
import org.apache.spark.sql.connector.write.Write;
import org.apache.spark.sql.connector.write.WriterCommitMessage;
import org.apache.spark.sql.connector.write.streaming.StreamingWrite;
import org.apache.spark.sql.execution.arrow.ArrowWriter;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.ArrowUtils;

/**
 * Spark write.
 */
public class SparkWrite implements Write {
  private final String datasetUri;
  private final StructType  sparkSchema;
  private final LogicalWriteInfo info;
  
  SparkWrite(String datasetUri, StructType sparkSchema, LogicalWriteInfo info) {
    this.datasetUri = datasetUri;
    this.sparkSchema = sparkSchema;
    this.info = info;
  }

  @Override
  public BatchWrite toBatch() {
    return new BatchAppend();
  }

  @Override
  public StreamingWrite toStreaming() {
    throw new UnsupportedOperationException();
  }

  private WriterFactory createWriterFactory() {
    return new WriterFactory(datasetUri, sparkSchema);
  }

  private class BatchAppend extends BaseBatchWrite {
    @Override
    public void commit(WriterCommitMessage[] messages) {
      List<FragmentMetadata> fragments = Arrays.stream(messages)
          .map(m -> (TaskCommit) m)
          .map(TaskCommit::getFragments)
          .flatMap(List::stream)
          .collect(Collectors.toList());
      FragmentOperation.Append appendOp = new FragmentOperation.Append(fragments);
      try (BufferAllocator allocator = SparkCatalog.newChildAllocator(
          "batch append commit", 0, Long.MAX_VALUE);
          Dataset datasetRead = Dataset.open(datasetUri, allocator)) {
        Dataset.commit(allocator, datasetUri, appendOp, Optional.of(datasetRead.version()));
      }
    }
  }

  private abstract class BaseBatchWrite implements BatchWrite {
    @Override
    public DataWriterFactory createBatchWriterFactory(PhysicalWriteInfo info) {
      return createWriterFactory();
    }

    @Override
    public boolean useCommitCoordinator() {
      return false;
    }

    @Override
    public void abort(WriterCommitMessage[] messages) {
      throw new UnsupportedOperationException();
    }

    @Override
    public String toString() {
      return String.format("LanceBatchWrite(datasetUri=%s)", datasetUri);
    }
  }

  private static class WriterFactory implements DataWriterFactory {
    private final String datasetUri;
    private final StructType sparkSchema;

    protected WriterFactory(String datasetUri, StructType sparkSchema) {
      // Everything passed to writer factory should be serializable
      this.datasetUri = datasetUri;
      this.sparkSchema = sparkSchema;
    }

    @Override
    public DataWriter<InternalRow> createWriter(int partitionId, long taskId) {
      return new UnpartitionedDataWriter(datasetUri, sparkSchema);
    }
  }

  private static class UnpartitionedDataWriter implements DataWriter<InternalRow> {
    private final String datasetUri;
    private final BufferAllocator allocator;
    private final VectorSchemaRoot root;
    private final ArrowWriter writer;
    
    private UnpartitionedDataWriter(String datasetUri, StructType sparkSchema) {
      this.datasetUri = datasetUri;
      this.allocator = SparkCatalog.newChildAllocator(
          "unpartitioned data writer", 0, Long.MAX_VALUE);
      root = VectorSchemaRoot.create(
          ArrowUtils.toArrowSchema(sparkSchema, datasetUri, true, false), allocator);
      writer = ArrowWriter.create(root);
    }

    @Override
    public void write(InternalRow record) {
      writer.write(record);
    }

    @Override
    public WriterCommitMessage commit() {
      writer.finish();
      return new TaskCommit(Arrays.asList(
          Fragment.create(datasetUri, allocator, root,
              Optional.empty(), new WriteParams.Builder().build())));
    }

    @Override
    public void abort() {
      close();
    }

    @Override
    public void close() {
      writer.reset();
      root.close();
      allocator.close();
    }
  }

  /** Task commit. */
  public static class TaskCommit implements WriterCommitMessage {
    private final List<FragmentMetadata> fragments;

    TaskCommit(List<FragmentMetadata> fragments) {
      this.fragments = fragments;
    }

    List<FragmentMetadata> getFragments() {
      return fragments;
    }
  }
}
