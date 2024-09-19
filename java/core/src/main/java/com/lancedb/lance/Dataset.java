/*
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

package com.lancedb.lance;

import com.lancedb.lance.index.IndexParams;
import com.lancedb.lance.index.IndexType;
import com.lancedb.lance.ipc.LanceScanner;
import com.lancedb.lance.ipc.ScanOptions;
import java.io.Closeable;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.util.Preconditions;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * Class representing a Lance dataset, interfacing with the native lance library. This class
 * provides functionality to open and manage datasets with native code. The native library is loaded
 * statically and utilized through native methods. It implements the {@link java.io.Closeable}
 * interface to ensure proper resource management.
 */
public class Dataset implements Closeable {
  static {
    JniLoader.ensureLoaded();
  }

  private long nativeDatasetHandle;

  BufferAllocator allocator;
  boolean selfManagedAllocator = false;

  private final LockManager lockManager = new LockManager();

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
    Preconditions.checkNotNull(allocator);
    Preconditions.checkNotNull(path);
    Preconditions.checkNotNull(schema);
    Preconditions.checkNotNull(params);
    try (ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator)) {
      Data.exportSchema(allocator, schema, null, arrowSchema);
      Dataset dataset =
          createWithFfiSchema(arrowSchema.memoryAddress(), path, params.getMaxRowsPerFile(),
              params.getMaxRowsPerGroup(), params.getMaxBytesPerFile(), params.getMode());
      dataset.allocator = allocator;
      return dataset;
    }
  }

  /**
   * Create a dataset with given stream.
   *
   * @param allocator buffer allocator
   * @param stream arrow stream
   * @param path dataset uri
   * @param params write parameters
   * @return Dataset
   */
  public static Dataset create(BufferAllocator allocator, ArrowArrayStream stream, String path,
      WriteParams params) {
    Preconditions.checkNotNull(allocator);
    Preconditions.checkNotNull(stream);
    Preconditions.checkNotNull(path);
    Preconditions.checkNotNull(params);
    Dataset dataset = createWithFfiStream(stream.memoryAddress(), path, params.getMaxRowsPerFile(),
        params.getMaxRowsPerGroup(), params.getMaxBytesPerFile(), params.getMode());
    dataset.allocator = allocator;
    return dataset;
  }

  private static native Dataset createWithFfiSchema(long arrowSchemaMemoryAddress, String path,
      Optional<Integer> maxRowsPerFile, Optional<Integer> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile, Optional<String> mode);

  private static native Dataset createWithFfiStream(long arrowStreamMemoryAddress, String path,
      Optional<Integer> maxRowsPerFile, Optional<Integer> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile, Optional<String> mode);

  /**
   * Open a dataset from the specified path.
   *
   * @param path file path
   * @return Dataset
   */
  public static Dataset open(String path) {
    return open(new RootAllocator(Long.MAX_VALUE), true, path, new ReadOptions.Builder().build());
  }

  /**
   * Open a dataset from the specified path.
   *
   * @param path file path
   * @param options the open options
   * @return Dataset
   */
  public static Dataset open(String path, ReadOptions options) {
    return open(new RootAllocator(Long.MAX_VALUE), true, path, options);
  }

  /**
   * Open a dataset from the specified path.
   *
   * @param path file path
   * @param allocator Arrow buffer allocator
   * @return Dataset
   */
  public static Dataset open(String path, BufferAllocator allocator) {
    return open(allocator, path, new ReadOptions.Builder().build());
  }

  /**
   * Open a dataset from the specified path with additional options.
   *
   * @param allocator Arrow buffer allocator
   * @param path file path
   * @param options the open options
   * @return Dataset
   */
  public static Dataset open(BufferAllocator allocator, String path, ReadOptions options) {
    return open(allocator, false, path, options);
  }

  /**
   * Open a dataset from the specified path with additional options.
   *
   * @param path file path
   * @param options the open options
   * @return Dataset
   */
  private static Dataset open(BufferAllocator allocator, boolean selfManagedAllocator, String path,
      ReadOptions options) {
    Preconditions.checkNotNull(path);
    Preconditions.checkNotNull(allocator);
    Preconditions.checkNotNull(options);
    Dataset dataset = openNative(path, options.getVersion(), options.getBlockSize(),
        options.getIndexCacheSize(), options.getMetadataCacheSize(), options.getStorageOptions());
    dataset.allocator = allocator;
    dataset.selfManagedAllocator = selfManagedAllocator;
    return dataset;
  }

  private static native Dataset openNative(String path, Optional<Integer> version,
      Optional<Integer> blockSize, int indexCacheSize, int metadataCacheSize,
      Map<String, String> storageOptions);

  /**
   * Create a new version of dataset.
   *
   * @param allocator the buffer allocator
   * @param path The file path of the dataset to open.
   * @param operation The operation to apply to the dataset.
   * @param readVersion The version of the dataset that was used as the base for the changes. This
   *        is not needed for overwrite or restore operations.
   * @return A new instance of {@link Dataset} linked to the opened dataset.
   */
  public static Dataset commit(BufferAllocator allocator, String path, FragmentOperation operation,
      Optional<Long> readVersion) {
    Preconditions.checkNotNull(allocator);
    Preconditions.checkNotNull(path);
    Preconditions.checkNotNull(operation);
    Preconditions.checkNotNull(readVersion);
    Dataset dataset = operation.commit(allocator, path, readVersion);
    dataset.allocator = allocator;
    return dataset;
  }

  public static native Dataset commitAppend(String path, Optional<Long> readVersion,
      List<String> fragmentsMetadata);

  /**
   * Create a new Dataset Scanner.
   *
   * @return a dataset scanner
   */
  public LanceScanner newScan() {
    return newScan(new ScanOptions.Builder().build());
  }

  /**
   * Create a new Dataset Scanner.
   *
   * @param batchSize the scan options with batch size, columns filter, and substrait
   * @return a dataset scanner
   */
  public LanceScanner newScan(long batchSize) {
    return newScan(new ScanOptions.Builder().batchSize(batchSize).build());
  }

  /**
   * Create a new Dataset Scanner.
   *
   * @param options the scan options
   * @return a dataset scanner
   */
  public LanceScanner newScan(ScanOptions options) {
    Preconditions.checkNotNull(options);
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return LanceScanner.create(this, options, allocator);
    }
  }

  /**
   * Gets the currently checked out version of the dataset.
   *
   * @return the version of the dataset
   */
  public long version() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeVersion();
    }
  }

  private native long nativeVersion();

  /**
   * @return the latest version of the dataset.
   */
  public long latestVersion() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeLatestVersion();
    }
  }

  private native long nativeLatestVersion();

  /**
   * Creates a new index on the dataset. Only vector indexes are supported.
   *
   * @param columns the columns to index from
   * @param indexType the index type
   * @param name the name of the created index
   * @param params index params
   * @param replace whether to replace the existing index
   */
  public void createIndex(List<String> columns, IndexType indexType, Optional<String> name,
      IndexParams params, boolean replace) {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeCreateIndex(columns, indexType.getValue(), name, params, replace);
    }
  }

  private native void nativeCreateIndex(List<String> columns, int indexTypeCode,
      Optional<String> name, IndexParams params, boolean replace);

  /**
   * Count the number of rows in the dataset.
   *
   * @return num of rows
   */
  public int countRows() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeCountRows();
    }
  }

  private native int nativeCountRows();

  /**
   * Get all fragments in this dataset.
   *
   * @return A list of {@link DatasetFragment}.
   */
  public List<DatasetFragment> getFragments() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      // Set a pointer in Fragment to dataset, to make it is easier to issue IOs
      // later.
      //
      // We do not need to close Fragments.
      return this.getJsonFragments().stream()
          .map(jsonFragment -> new DatasetFragment(this, FragmentMetadata.fromJson(jsonFragment)))
          .collect(Collectors.toList());
    }
  }

  private native List<String> getJsonFragments();

  /**
   * Gets the schema of the dataset.
   *
   * @return the arrow schema
   */
  public Schema getSchema() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      try (ArrowSchema ffiArrowSchema = ArrowSchema.allocateNew(allocator)) {
        importFfiSchema(ffiArrowSchema.memoryAddress());
        return Data.importSchema(allocator, ffiArrowSchema, null);
      }
    }
  }

  private native void importFfiSchema(long arrowSchemaMemoryAddress);

  /**
   * @return all the created indexes names
   */
  public List<String> listIndexes() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeListIndexes();
    }
  }

  private native List<String> nativeListIndexes();

  /**
   * Closes this dataset and releases any system resources associated with it. If the dataset is
   * already closed, then invoking this method has no effect.
   */
  @Override
  public void close() {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      if (nativeDatasetHandle != 0) {
        releaseNativeDataset(nativeDatasetHandle);
        nativeDatasetHandle = 0;
      }
      if (selfManagedAllocator) {
        allocator.close();
      }
    }
  }

  /**
   * Native method to release the Lance dataset resources associated with the given handle.
   *
   * @param handle The native handle to the dataset resource.
   */
  private native void releaseNativeDataset(long handle);

  /**
   * Checks if the dataset is closed.
   *
   * @return true if the dataset is closed, false otherwise.
   */
  public boolean closed() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      return nativeDatasetHandle == 0;
    }
  }
}
