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
package com.lancedb.lance.file;

import com.lancedb.lance.JniLoader;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Schema;

import javax.annotation.Nullable;

import java.io.IOException;
import java.util.List;

public class LanceFileReader implements AutoCloseable {

  static {
    JniLoader.ensureLoaded();
  }

  private long nativeFileReaderHandle;

  private BufferAllocator allocator;
  private Schema schema;

  private static native LanceFileReader openNative(String fileUri) throws IOException;

  private native void closeNative(long nativeLanceFileReaderHandle) throws IOException;

  private native long numRowsNative() throws IOException;

  private native void populateSchemaNative(long arrowSchemaMemoryAddress);

  private native void readAllNative(
      int batchSize,
      @Nullable List<String> projectedNames,
      @Nullable int[] ranges,
      long streamMemoryAddress)
      throws IOException;

  private LanceFileReader() {}

  /**
   * Open a LanceFileReader from a file URI
   *
   * @param path the URI to the Lance file
   * @param allocator the Arrow BufferAllocator to use for the reader
   * @return a new LanceFileReader
   */
  public static LanceFileReader open(String path, BufferAllocator allocator) throws IOException {
    LanceFileReader reader = openNative(path);
    reader.allocator = allocator;
    reader.schema = reader.load_schema();
    return reader;
  }

  /**
   * Close the LanceFileReader
   *
   * <p>This method must be called to release resources when the reader is no longer needed.
   */
  @Override
  public void close() throws IOException {
    closeNative(nativeFileReaderHandle);
  }

  /**
   * Get the number of rows in the Lance file
   *
   * @return the number of rows in the Lance file
   */
  public long numRows() throws IOException {
    long numRows = numRowsNative();
    return numRows;
  }

  /**
   * Get the schema of the Lance file
   *
   * @return the schema of the Lance file
   */
  public Schema schema() {
    return schema;
  }

  private Schema load_schema() throws IOException {
    try (ArrowSchema ffiArrowSchema = ArrowSchema.allocateNew(allocator)) {
      populateSchemaNative(ffiArrowSchema.memoryAddress());
      return Data.importSchema(allocator, ffiArrowSchema, null);
    }
  }

  /**
   * Read all rows from the Lance file
   *
   * @param batchSize the maximum number of rows to read in a single batch
   * @param projectedNames optional list of column names to project; if null, all columns are read
   * @param ranges optional array of ranges to read; if null, all rows are read. The array must be
   *     even, with pattern [start1, end1, start2, end2, ...], where each pair defines a range of
   *     row indices to read. The starts are inclusive and ends are exclusive.
   * @return an ArrowReader for the Lance file
   */
  public ArrowReader readAll(
      @Nullable List<String> projectedNames, @Nullable int[] ranges, int batchSize)
      throws IOException {
    if (ranges != null && ranges.length % 2 != 0) {
      throw new IllegalArgumentException("Ranges must be an even-length array.");
    }
    try (ArrowArrayStream ffiArrowArrayStream = ArrowArrayStream.allocateNew(allocator)) {
      readAllNative(batchSize, projectedNames, ranges, ffiArrowArrayStream.memoryAddress());
      return Data.importArrayStream(allocator, ffiArrowArrayStream);
    }
  }
}
