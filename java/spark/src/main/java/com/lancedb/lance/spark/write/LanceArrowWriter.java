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
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.execution.arrow.ArrowWriter;

import javax.annotation.concurrent.GuardedBy;
import java.io.IOException;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * A custom arrow reader that supports writes Spark internal rows while reading data in batches.
 */
public class LanceArrowWriter extends ArrowReader {
  private final Schema schema;
  private final int batchSize;
  private final Object monitor = new Object();
  @GuardedBy("monitor")
  private final Queue<InternalRow> rowQueue = new ConcurrentLinkedQueue<>();
  @GuardedBy("monitor")
  private volatile boolean finished;

  private final AtomicLong totalBytesRead = new AtomicLong();
  private ArrowWriter arrowWriter = null;
  private final AtomicInteger count = new AtomicInteger(0);
  private final Semaphore writeToken;
  private final Semaphore loadToken;

  public LanceArrowWriter(BufferAllocator allocator, Schema schema, int batchSize) {
    super(allocator);
    Preconditions.checkNotNull(schema);
    Preconditions.checkArgument(batchSize > 0);
    this.schema = schema;
    // TODO(lu) batch size as config?
    this.batchSize = batchSize;
    this.writeToken = new Semaphore(0);
    this.loadToken = new Semaphore(0);
  }

  void write(InternalRow row) {
    Preconditions.checkNotNull(row);
    try {
      // wait util prepareLoadNextBatch to release write token,
      writeToken.acquire();
      arrowWriter.write(row);
      if (count.incrementAndGet() == batchSize) {
        // notify loadNextBatch to take the batch
        loadToken.release();
      }
    } catch (InterruptedException e) {
        throw new RuntimeException(e);
    }
  }

  void setFinished() {
    loadToken.release();
    finished = true;
  }

  @Override
  public void prepareLoadNextBatch() throws IOException {
    super.prepareLoadNextBatch();
    arrowWriter = ArrowWriter.create(this.getVectorSchemaRoot());
    // release batch size token for write
    writeToken.release(batchSize);
  }

  @Override
  public boolean loadNextBatch() throws IOException {
    prepareLoadNextBatch();
    try {
      if (finished && count.get() == 0) {
        return false;
      }
      // wait util batch if full or finished
      loadToken.acquire();
      arrowWriter.finish();
      if (!finished) {
        count.set(0);
        return true;
      } else {
        // true if it has some rows and return false if there is no record
        if (count.get() > 0) {
          count.set(0);
          return true;
        } else {
          return false;
        }
      }
    } catch (InterruptedException e) {
        throw new RuntimeException(e);
    }
  }

  @Override
  public long bytesRead() {
    throw new UnsupportedOperationException();
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
