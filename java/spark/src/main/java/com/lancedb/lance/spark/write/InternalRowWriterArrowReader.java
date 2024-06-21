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

package com.lancedb.lance.spark.write;

import com.google.common.base.Preconditions;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.VectorUnloader;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.message.ArrowRecordBatch;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.execution.arrow.ArrowWriter;

import javax.annotation.concurrent.GuardedBy;
import java.io.IOException;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;

/**
 * A custom arrow reader that supports writes Spark internal rows while reading data in batches.
 */
public class InternalRowWriterArrowReader extends ArrowReader {
  private final Schema schema;
  private final int batchSize;
  private final Object monitor = new Object();
  @GuardedBy("monitor")
  private final Queue<InternalRow> rowQueue = new ConcurrentLinkedQueue<>();
  @GuardedBy("monitor")
  private volatile boolean finished;

  private final AtomicLong totalBytesRead = new AtomicLong();
  private ArrowWriter arrowWriter = null;

  public InternalRowWriterArrowReader(BufferAllocator allocator, Schema schema, int batchSize) {
    super(allocator);
    Preconditions.checkNotNull(schema);
    Preconditions.checkArgument(batchSize > 0);
    this.schema = schema;
    // TODO(lu) batch size as config?
    this.batchSize = batchSize;
  }

  public void write(InternalRow row) {
    Preconditions.checkNotNull(row);
    synchronized (monitor) {
      // TODO(lu) wait if too much elements in rowQueue
      rowQueue.offer(row);
      monitor.notify();
    }
  }

  public void setFinished() {
    synchronized (monitor) {
      finished = true;
      monitor.notify();
    }
  }

  @Override
  protected void prepareLoadNextBatch() throws IOException {
    super.prepareLoadNextBatch();
    // Do not use ArrowWriter.reset since it does not work well with Arrow JNI
    arrowWriter = ArrowWriter.create(this.getVectorSchemaRoot());
  }

  @Override
  public boolean loadNextBatch() throws IOException {
    prepareLoadNextBatch();
    int rowCount = 0;
    synchronized (monitor) {
      while (rowCount < batchSize) {
        while (rowQueue.isEmpty() && !finished) {
          try {
            monitor.wait();
          } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted while waiting for data", e);
          }
        }
        if (rowQueue.isEmpty() && finished) {
          break;
        }
        InternalRow row = rowQueue.poll();
        if (row != null) {
          arrowWriter.write(row);
          rowCount++;
        }
      }
    }
    if (rowCount == 0) {
      return false;
    }
    arrowWriter.finish();
    // Calculate bytes read for the current record batch
    // If the following code impacts performance, can be removed
    VectorSchemaRoot root = this.getVectorSchemaRoot();
    VectorUnloader unloader = new VectorUnloader(root);
    try (ArrowRecordBatch recordBatch = unloader.getRecordBatch()) {
      totalBytesRead.addAndGet(recordBatch.computeBodyLength());
    }
    return true;
  }

  @Override
  public long bytesRead() {
    return totalBytesRead.get();
  }

  @Override
  protected synchronized void closeReadSource() throws IOException {
    // Implement if needed
  }

  @Override
  protected Schema readSchema() {
    return this.schema;
  }
}
