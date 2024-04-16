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

import io.questdb.jar.jni.JarJniLoader;
import java.io.Closeable;
import java.io.IOException;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * Class representing a Lance dataset, interfacing with the native lance library. This class
 * provides functionality to open and manage datasets with native code. The native library is loaded
 * statically and utilized through native methods. It implements the {@link java.io.Closeable}
 * interface to ensure proper resource management.
 */
public class Dataset implements Closeable {
  static {
    JarJniLoader.loadLib(Dataset.class, "/nativelib", "lance_jni");
  }

  private long nativeDatasetHandle;

  BufferAllocator allocator;

  private Dataset() {}

  /**
   * Creates an empty dataset.
   *
   * @param allocator the buffer allocator
   * @param path dataset uri
   * @param schema dataset schema
   * @param params write params
   * @return Dataset
   */
  public static Dataset create(BufferAllocator allocator, String path, Schema schema,
      WriteParams params) {
    try (ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator)) {
      Data.exportSchema(allocator, schema, null, arrowSchema);
      var dataset =  createWithFfiSchema(arrowSchema.memoryAddress(),
          path, params.getMaxRowsPerFile(), params.getMaxRowsPerGroup(),
          params.getMaxBytesPerFile(), params.getMode());
      dataset.allocator = allocator;
      return dataset;
    }
  }

  private static native Dataset createWithFfiSchema(long arrowSchemaMemoryAddress, String path,
      Optional<Integer> maxRowsPerFile, Optional<Integer> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile, Optional<String> mode);

  /**
   * Write a dataset to the specified path.
   *
   * @param allocator buffer allocator
   * @param stream arrow stream
   * @param path dataset uri
   * @param params write parameters
   * @return Dataset
   */
  public static Dataset write(BufferAllocator allocator, ArrowArrayStream stream,
      String path, WriteParams params) {
    var dataset = writeWithFfiStream(stream.memoryAddress(), path,
        params.getMaxRowsPerFile(), params.getMaxRowsPerGroup(),
        params.getMaxBytesPerFile(), params.getMode());
    dataset.allocator = allocator;
    return dataset;
  }

  private static native Dataset writeWithFfiStream(long arrowStreamMemoryAddress, String path,
      Optional<Integer> maxRowsPerFile, Optional<Integer> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile, Optional<String> mode);

  /**
   * Open a dataset from the specified path.
   *
   * @param path file path
   * @param allocator Arrow buffer allocator.
   * @return Dataset
   */
  public static Dataset open(String path, BufferAllocator allocator) throws IOException {
    var dataset = openNative(path);
    dataset.allocator = allocator;
    return dataset;
  }

  /**
   * Opens a dataset from the specified path using the native library.
   *
   * @param path The file path of the dataset to open.
   * @return A new instance of {@link Dataset} linked to the opened dataset.
   */
  public static native Dataset openNative(String path);

  /**
   * Opens a dataset from the specified path using the native library.
   *
   * @param path The file path of the dataset to open.
   * @return A new instance of {@link Dataset} linked to the opened dataset.
   */
  public static Dataset commit(BufferAllocator allocator, String path,
      FragmentOperation operation, Optional<Integer> readVersion) {
    var dataset = operation.commit(allocator, path, readVersion);
    dataset.allocator = allocator;
    return dataset;
  }

  public static native Dataset commitAppend(String path, Optional<Integer> readVersion,
      List<String> fragmentsMetadata);

  /**
   * Count the number of rows in the dataset.
   *
   * @return num of rows.
   */
  public native int countRows();

  /**
   * Get all fragments in this dataset.
   *
   * @return A list of {@link DatasetFragment}.
   */
  public List<DatasetFragment> getFragments() {
    // Set a pointer in Fragment to dataset, to make it is easier to issue IOs later.
    //
    // We do not need to close Fragments.
    return this.getJsonFragments().stream().map(jsonFragment 
        -> new DatasetFragment(this, new FragmentMetadata(jsonFragment)))
        .collect(Collectors.toList());
  }

  private native List<String> getJsonFragments();

  /**
   * Gets the schema of the dataset.
   *
   * @return the arrow schema
   */
  public Schema getSchema() {
    try (ArrowSchema ffiArrowSchema = ArrowSchema.allocateNew(allocator)) {
      importFfiSchema(ffiArrowSchema.memoryAddress());
      return Data.importSchema(allocator, ffiArrowSchema, null);
    }
  }

  private native void importFfiSchema(long arrowSchemaMemoryAddress);

  /**
   * Closes this dataset and releases any system resources associated with it. If the dataset is
   * already closed, then invoking this method has no effect.
   */
  @Override
  public void close() {
    if (nativeDatasetHandle != 0) {
      releaseNativeDataset(nativeDatasetHandle);
      nativeDatasetHandle = 0;
    }
  }

  /**
   * Native method to release the Lance dataset resources associated with the given handle.
   *
   * @param handle The native handle to the dataset resource.
   */
  private native void releaseNativeDataset(long handle);
}
