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
package org.lance.memwal;

import org.lance.JniLoader;
import org.lance.LockManager;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.util.Preconditions;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * An executable physical plan produced by a MemWAL planner.
 *
 * <p>Planner classes only build plans; execution happens through this class. Obtain instances from
 * {@link LsmPointLookupPlanner#planLookup} or {@link LsmVectorSearchPlanner#planSearch}.
 */
public class ExecutionPlan implements AutoCloseable {
  static {
    JniLoader.ensureLoaded();
  }

  private long nativeExecutionPlanHandle;
  BufferAllocator allocator;
  private final LockManager lockManager = new LockManager();

  private ExecutionPlan() {}

  /** Output schema of this physical plan. */
  public Schema schema() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeExecutionPlanHandle != 0, "ExecutionPlan is closed");
      try (ArrowSchema ffiSchema = ArrowSchema.allocateNew(allocator)) {
        nativeImportSchema(ffiSchema.memoryAddress());
        return Data.importSchema(allocator, ffiSchema, null);
      }
    }
  }

  private native void nativeImportSchema(long schemaAddress);

  /** Schema of the base dataset this plan was constructed from. */
  public Schema datasetSchema() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeExecutionPlanHandle != 0, "ExecutionPlan is closed");
      try (ArrowSchema ffiSchema = ArrowSchema.allocateNew(allocator)) {
        nativeImportDatasetSchema(ffiSchema.memoryAddress());
        return Data.importSchema(allocator, ffiSchema, null);
      }
    }
  }

  private native void nativeImportDatasetSchema(long schemaAddress);

  /** Return the physical plan rendered as an indented string. */
  public String explain() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeExecutionPlanHandle != 0, "ExecutionPlan is closed");
      return nativeExplain();
    }
  }

  private native String nativeExplain();

  /** Execute the plan and return a streaming reader over the results. */
  public ArrowReader toReader() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeExecutionPlanHandle != 0, "ExecutionPlan is closed");
      try (ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
        nativeOpenStream(stream.memoryAddress());
        return Data.importArrayStream(allocator, stream);
      }
    }
  }

  private native void nativeOpenStream(long streamAddress);

  /**
   * Close the plan and release native resources. If the plan is already closed, invoking this
   * method has no effect.
   */
  @Override
  public void close() {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      if (nativeExecutionPlanHandle != 0) {
        releaseNativeExecutionPlan(nativeExecutionPlanHandle);
        nativeExecutionPlanHandle = 0;
      }
    }
  }

  private native void releaseNativeExecutionPlan(long handle);
}
