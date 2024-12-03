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

package com.lancedb.lance.ipc;

import com.lancedb.lance.Dataset;
import com.lancedb.lance.LockManager;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.dataset.scanner.ScanTask;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.util.Preconditions;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Schema;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Optional;

/** Scanner over a Fragment. */
public class LanceScanner implements org.apache.arrow.dataset.scanner.Scanner {
  Dataset dataset;

  ScanOptions options;

  BufferAllocator allocator;

  private long nativeScannerHandle;

  private final LockManager lockManager = new LockManager();

  private LanceScanner() {}

  /**
   * Create a Scanner.
   *
   * @param dataset the dataset to scan
   * @param options scan options
   * @param allocator allocator
   * @return a Scanner
   */
  public static LanceScanner create(
      Dataset dataset, ScanOptions options, BufferAllocator allocator) {
    Preconditions.checkNotNull(dataset);
    Preconditions.checkNotNull(options);
    Preconditions.checkNotNull(allocator);
    LanceScanner scanner =
        createScanner(
            dataset,
            options.getFragmentIds(),
            options.getColumns(),
            options.getSubstraitFilter(),
            options.getFilter(),
            options.getBatchSize(),
            options.getLimit(),
            options.getOffset(),
            options.getNearest(),
            options.isWithRowId(),
            options.getBatchReadahead());
    scanner.allocator = allocator;
    scanner.dataset = dataset;
    scanner.options = options;
    return scanner;
  }

  static native LanceScanner createScanner(
      Dataset dataset,
      Optional<List<Integer>> fragmentIds,
      Optional<List<String>> columns,
      Optional<ByteBuffer> substraitFilter,
      Optional<String> filter,
      Optional<Long> batchSize,
      Optional<Long> limit,
      Optional<Long> offset,
      Optional<Query> query,
      boolean withRowId,
      int batchReadahead);

  /**
   * Closes this scanner and releases any system resources associated with it. If the scanner is
   * already closed, then invoking this method has no effect.
   */
  @Override
  public void close() throws Exception {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      if (nativeScannerHandle != 0) {
        releaseNativeScanner(nativeScannerHandle);
        nativeScannerHandle = 0;
      }
    }
  }

  /**
   * Native method to release the Lance scanner resources associated with the given handle.
   *
   * @param handle The native handle to the scanner resource.
   */
  private native void releaseNativeScanner(long handle);

  @Override
  public ArrowReader scanBatches() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeScannerHandle != 0, "Scanner is closed");
      try (ArrowArrayStream s = ArrowArrayStream.allocateNew(allocator)) {
        openStream(s.memoryAddress());
        return Data.importArrayStream(allocator, s);
      } catch (IOException e) {
        // TODO: handle IO exception?
        throw new RuntimeException(e);
      }
    }
  }

  private native void openStream(long streamAddress) throws IOException;

  @Override
  public Iterable<? extends ScanTask> scan() {
    // Marked as deprecated in Scanner
    throw new UnsupportedOperationException();
  }

  @Override
  public Schema schema() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeScannerHandle != 0, "Scanner is closed");
      try (ArrowSchema ffiArrowSchema = ArrowSchema.allocateNew(allocator)) {
        importFfiSchema(ffiArrowSchema.memoryAddress());
        return Data.importSchema(allocator, ffiArrowSchema, null);
      }
    }
  }

  private native void importFfiSchema(long arrowSchemaMemoryAddress);

  /**
   * Scan and return the number of matching rows (fulfill the given scan options).
   *
   * @return num of rows.
   */
  public long countRows() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeScannerHandle != 0, "Scanner is closed");
      return nativeCountRows();
    }
  }

  private native long nativeCountRows();
}
