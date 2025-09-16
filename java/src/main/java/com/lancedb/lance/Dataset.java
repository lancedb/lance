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

import com.lancedb.lance.index.IndexParams;
import com.lancedb.lance.index.IndexType;
import com.lancedb.lance.ipc.DataStatistics;
import com.lancedb.lance.ipc.LanceScanner;
import com.lancedb.lance.ipc.ScanOptions;
import com.lancedb.lance.merge.MergeInsertParams;
import com.lancedb.lance.merge.MergeInsertResult;
import com.lancedb.lance.schema.ColumnAlteration;
import com.lancedb.lance.schema.LanceSchema;
import com.lancedb.lance.schema.SqlExpressions;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.util.Preconditions;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;

import java.io.ByteArrayInputStream;
import java.io.Closeable;
import java.io.IOException;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

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

  private BufferAllocator allocator;
  private boolean selfManagedAllocator = false;

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
  public static Dataset create(
      BufferAllocator allocator, String path, Schema schema, WriteParams params) {
    Preconditions.checkNotNull(allocator);
    Preconditions.checkNotNull(path);
    Preconditions.checkNotNull(schema);
    Preconditions.checkNotNull(params);
    try (ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator)) {
      Data.exportSchema(allocator, schema, null, arrowSchema);
      Dataset dataset =
          createWithFfiSchema(
              arrowSchema.memoryAddress(),
              path,
              params.getMaxRowsPerFile(),
              params.getMaxRowsPerGroup(),
              params.getMaxBytesPerFile(),
              params.getMode(),
              params.getEnableStableRowIds(),
              params.getStorageOptions());
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
  public static Dataset create(
      BufferAllocator allocator, ArrowArrayStream stream, String path, WriteParams params) {
    Preconditions.checkNotNull(allocator);
    Preconditions.checkNotNull(stream);
    Preconditions.checkNotNull(path);
    Preconditions.checkNotNull(params);
    Dataset dataset =
        createWithFfiStream(
            stream.memoryAddress(),
            path,
            params.getMaxRowsPerFile(),
            params.getMaxRowsPerGroup(),
            params.getMaxBytesPerFile(),
            params.getMode(),
            params.getEnableStableRowIds(),
            params.getStorageOptions());
    dataset.allocator = allocator;
    return dataset;
  }

  private static native Dataset createWithFfiSchema(
      long arrowSchemaMemoryAddress,
      String path,
      Optional<Integer> maxRowsPerFile,
      Optional<Integer> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile,
      Optional<String> mode,
      Optional<Boolean> enableStableRowIds,
      Map<String, String> storageOptions);

  private static native Dataset createWithFfiStream(
      long arrowStreamMemoryAddress,
      String path,
      Optional<Integer> maxRowsPerFile,
      Optional<Integer> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile,
      Optional<String> mode,
      Optional<Boolean> enableStableRowIds,
      Map<String, String> storageOptions);

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
  private static Dataset open(
      BufferAllocator allocator, boolean selfManagedAllocator, String path, ReadOptions options) {
    Preconditions.checkNotNull(path);
    Preconditions.checkNotNull(allocator);
    Preconditions.checkNotNull(options);
    Dataset dataset =
        openNative(
            path,
            options.getVersion(),
            options.getBlockSize(),
            options.getIndexCacheSizeBytes(),
            options.getMetadataCacheSizeBytes(),
            options.getStorageOptions());
    dataset.allocator = allocator;
    dataset.selfManagedAllocator = selfManagedAllocator;
    return dataset;
  }

  private static native Dataset openNative(
      String path,
      Optional<Integer> version,
      Optional<Integer> blockSize,
      long indexCacheSize,
      long metadataCacheSizeBytes,
      Map<String, String> storageOptions);

  /**
   * Create a new version of dataset. Use {@link Transaction} instead
   *
   * @param allocator the buffer allocator
   * @param path The file path of the dataset to open.
   * @param operation The operation to apply to the dataset.
   * @param readVersion The version of the dataset that was used as the base for the changes. This
   *     is not needed for overwrite or restore operations.
   * @return A new instance of {@link Dataset} linked to the opened dataset.
   */
  @Deprecated
  public static Dataset commit(
      BufferAllocator allocator,
      String path,
      FragmentOperation operation,
      Optional<Long> readVersion) {
    return commit(allocator, path, operation, readVersion, new HashMap<>());
  }

  @Deprecated
  public static Dataset commit(
      BufferAllocator allocator,
      String path,
      FragmentOperation operation,
      Optional<Long> readVersion,
      Map<String, String> storageOptions) {
    Preconditions.checkNotNull(allocator);
    Preconditions.checkNotNull(path);
    Preconditions.checkNotNull(operation);
    Preconditions.checkNotNull(readVersion);
    Dataset dataset = operation.commit(allocator, path, readVersion, storageOptions);
    dataset.allocator = allocator;
    return dataset;
  }

  /** Use {@link Transaction} instead */
  @Deprecated
  public static native Dataset commitAppend(
      String path,
      Optional<Long> readVersion,
      List<FragmentMetadata> fragmentsMetadata,
      Map<String, String> storageOptions);

  /** Use {@link Transaction} instead */
  @Deprecated
  public static native Dataset commitOverwrite(
      String path,
      long arrowSchemaMemoryAddress,
      Optional<Long> readVersion,
      List<FragmentMetadata> fragmentsMetadata,
      Map<String, String> storageOptions);

  public BufferAllocator allocator() {
    return allocator;
  }

  /**
   * Create a new transaction builder at current version for the dataset. The dataset itself will
   * not refresh after the transaction committed.
   *
   * @return A new instance of {@link Transaction.Builder} linked to the opened dataset.
   */
  public Transaction.Builder newTransactionBuilder() {
    return new Transaction.Builder(this).readVersion(version());
  }

  /**
   * Commit a single transaction and return a new Dataset with the new version. Original dataset
   * version will not be refreshed.
   *
   * @param transaction The transaction to commit
   * @return A new instance of {@link Dataset} linked to committed version.
   */
  public Dataset commitTransaction(Transaction transaction) {
    Preconditions.checkNotNull(transaction);
    try {
      Dataset dataset = nativeCommitTransaction(transaction);
      if (selfManagedAllocator) {
        dataset.allocator = new RootAllocator(Long.MAX_VALUE);
      } else {
        dataset.allocator = allocator;
      }
      return dataset;
    } finally {
      transaction.release();
    }
  }

  private native Dataset nativeCommitTransaction(Transaction transaction);

  /**
   * Drop a Dataset.
   *
   * @param path The file path of the dataset
   * @param storageOptions Storage options
   */
  public static native void drop(String path, Map<String, String> storageOptions);

  /**
   * Add columns to the dataset.
   *
   * @param sqlExpressions The SQL expressions to add columns
   * @param batchSize The number of rows to read at a time from the source dataset when applying the
   *     transform.
   */
  public void addColumns(SqlExpressions sqlExpressions, Optional<Long> batchSize) {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeAddColumnsBySqlExpressions(sqlExpressions, batchSize);
    }
  }

  private native void nativeAddColumnsBySqlExpressions(
      SqlExpressions sqlExpressions, Optional<Long> batchSize);

  /**
   * Add columns to the dataset.
   *
   * @param stream The Arrow Array Stream generated by arrow reader to add columns.
   * @param batchSize The number of rows to read at a time from the source dataset when applying the
   *     transform.
   */
  public void addColumns(ArrowArrayStream stream, Optional<Long> batchSize) {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeAddColumnsByReader(stream.memoryAddress(), batchSize);
    }
  }

  private native void nativeAddColumnsByReader(
      long arrowStreamMemoryAddress, Optional<Long> batchSize);

  /**
   * Add columns to the dataset.
   *
   * @param schema The Arrow schema definitions to add columns.
   */
  public void addColumns(Schema schema) {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      Preconditions.checkArgument(schema != null, "Schema is empty");
      try (ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator)) {
        Data.exportSchema(allocator, schema, null, arrowSchema);
        nativeAddColumnsBySchema(arrowSchema.memoryAddress());
      }
    }
  }

  /**
   * Add columns to the dataset.
   *
   * @param fields The Arrow field definitions to add columns.
   */
  public void addColumns(List<Field> fields) {
    Preconditions.checkArgument(fields != null && !fields.isEmpty(), "Fields are empty");
    addColumns(new Schema(fields));
  }

  private native void nativeAddColumnsBySchema(long schemaPtr);

  /**
   * Drop columns from the dataset.
   *
   * @param columns The columns to drop
   */
  public void dropColumns(List<String> columns) {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeDropColumns(columns);
    }
  }

  private native void nativeDropColumns(List<String> columns);

  /**
   * Alter columns in the dataset.
   *
   * @param columnAlterations The list of columns need to be altered.
   */
  public void alterColumns(List<ColumnAlteration> columnAlterations) {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeAlterColumns(columnAlterations);
    }
  }

  private native void nativeAlterColumns(List<ColumnAlteration> columnAlterations);

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
   * Select rows of data by index.
   *
   * @param indices the indices to take
   * @param columns the columns to take
   * @return an ArrowReader
   */
  public ArrowReader take(List<Long> indices, List<String> columns) throws IOException {
    Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      byte[] arrowData = nativeTake(indices, columns);
      ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(arrowData);
      ReadableByteChannel readChannel = Channels.newChannel(byteArrayInputStream);
      return new ArrowStreamReader(readChannel, allocator) {
        @Override
        public void close() throws IOException {
          super.close();
          readChannel.close();
          byteArrayInputStream.close();
        }
      };
    }
  }

  private native byte[] nativeTake(List<Long> indices, List<String> columns);

  /**
   * Delete rows of data by predicate.
   *
   * @param predicate the predicate to delete
   */
  public void delete(String predicate) {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeDelete(predicate);
    }
  }

  private native void nativeDelete(String predicate);

  /**
   * Gets the URI of the dataset.
   *
   * @return the URI of the dataset
   */
  public String uri() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeUri();
    }
  }

  private native String nativeUri();

  /**
   * Get the currently checked out version id of the dataset
   *
   * @return the version id of the dataset
   */
  public long version() {
    return getVersion().getId();
  }

  /**
   * Gets the currently checked out version of the dataset.
   *
   * @return the version of the dataset
   */
  public Version getVersion() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeGetVersion();
    }
  }

  private native Version nativeGetVersion();

  /**
   * Get the version history of the dataset.
   *
   * @return the version history of the dataset
   */
  public List<Version> listVersions() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeListVersions();
    }
  }

  private native List<Version> nativeListVersions();

  /** @return the latest version of the dataset. */
  public long latestVersion() {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeGetLatestVersionId();
    }
  }

  private native long nativeGetLatestVersionId();

  /** Checkout the dataset to the latest version. */
  public void checkoutLatest() {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeCheckoutLatest();
    }
  }

  private native void nativeCheckoutLatest();

  /**
   * Checks out a specific version of the dataset. If the version is already checked out, it returns
   * a new Java Dataset object pointing to the same underlying Rust Dataset object
   *
   * @param version the version to check out
   * @return a new Dataset instance with the specified version checked out
   */
  public Dataset checkoutVersion(long version) {
    Preconditions.checkArgument(version > 0, "version number must be greater than 0");
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeCheckoutVersion(version);
    }
  }

  private native Dataset nativeCheckoutVersion(long version);

  /**
   * Checks out a specific tag of the dataset. If the underlying version is already checked out, it
   * returns a new Java Dataset object pointing to the same underlying Rust Dataset object
   *
   * @param tag the tag to check out
   * @return a new Dataset instance with the specified tag checked out
   */
  public Dataset checkoutTag(String tag) {
    Preconditions.checkArgument(tag != null, "Tag can not be null");
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeCheckoutTag(tag);
    }
  }

  private native Dataset nativeCheckoutTag(String tag);

  /**
   * Restore the currently checked out version of the dataset as the latest version. This operation
   * produces a new version and doesn't influence any old versions and tags.
   */
  public void restore() {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeRestore();
    }
  }

  private native void nativeRestore();

  /**
   * Creates a new index on the dataset. Only vector indexes are supported.
   *
   * @param columns the columns to index from
   * @param indexType the index type
   * @param name the name of the created index
   * @param params index params
   * @param replace whether to replace the existing index
   */
  public void createIndex(
      List<String> columns,
      IndexType indexType,
      Optional<String> name,
      IndexParams params,
      boolean replace) {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeCreateIndex(columns, indexType.getValue(), name, params, replace);
    }
  }

  private native void nativeCreateIndex(
      List<String> columns,
      int indexTypeCode,
      Optional<String> name,
      IndexParams params,
      boolean replace);

  /**
   * Count the number of rows in the dataset.
   *
   * @return num of rows
   */
  public long countRows() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeCountRows(Optional.empty());
    }
  }

  /**
   * Count the number of rows in the dataset.
   *
   * @param filter the filter expr to count row
   * @return num of rows
   */
  public long countRows(String filter) {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      Preconditions.checkArgument(
          null != filter && !filter.isEmpty(), "filter cannot be null or empty");
      return nativeCountRows(Optional.of(filter));
    }
  }

  private native long nativeCountRows(Optional<String> filter);

  /**
   * Calculate the size of the dataset.
   *
   * @return the size of the dataset
   */
  public long calculateDataSize() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeGetDataStatistics().getDataSize();
    }
  }

  /**
   * Calculate the statistics of the dataset.
   *
   * @return the statistics of the dataset
   */
  private native DataStatistics nativeGetDataStatistics();

  /**
   * Get all fragments in this dataset.
   *
   * @return A list of {@link Fragment}.
   */
  public List<Fragment> getFragments() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      // Set a pointer in Fragment to dataset, to make it is easier to issue IOs
      // later.
      //
      // We do not need to close Fragments.
      return this.getFragmentsNative().stream()
          .map(metadata -> new Fragment(this, metadata))
          .collect(Collectors.toList());
    }
  }

  private native List<FragmentMetadata> getFragmentsNative();

  /**
   * Gets the arrow schema of the dataset.
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
   * Get the {@link com.lancedb.lance.schema.LanceSchema} of the dataset with field ids.
   *
   * @return the LanceSchema
   */
  public LanceSchema getLanceSchema() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeGetLanceSchema();
    }
  }

  private native LanceSchema nativeGetLanceSchema();

  /**
   * Get the {@link com.lancedb.lance.Transaction} of the dataset at the current version.
   *
   * @return the Transaction
   */
  public Optional<Transaction> readTransaction() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return Optional.ofNullable(nativeReadTransaction());
    }
  }

  private native Transaction nativeReadTransaction();

  /** @return all the created indexes names */
  public List<String> listIndexes() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeListIndexes();
    }
  }

  private native List<String> nativeListIndexes();

  /**
   * Get the table config of the dataset.
   *
   * @return the table config
   */
  public Map<String, String> getConfig() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeGetConfig();
    }
  }

  private native Map<String, String> nativeGetConfig();

  /**
   * Compact the dataset to improve performance.
   *
   * <p>This operation performs several optimizations:
   *
   * <ul>
   *   <li>Removes deleted rows from fragments
   *   <li>Removes dropped columns from fragments
   *   <li>Merges fragments that are too small
   * </ul>
   *
   * @param options compaction options to control the behavior
   */
  public void compact(CompactionOptions options) {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeCompact(options);
    }
  }

  /** Compact the dataset with default options. */
  public void compact() {
    compact(CompactionOptions.builder().build());
  }

  private native void nativeCompact(CompactionOptions options);

  /**
   * Update the config of the dataset. This operation will only overwrite and NOT delete the
   * existing config.
   *
   * @param tableConfig the config to update
   */
  public void updateConfig(Map<String, String> tableConfig) {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeUpdateConfig(tableConfig);
    }
  }

  private native void nativeUpdateConfig(Map<String, String> config);

  /**
   * Delete the config keys of the dataset.
   *
   * @param deleteKeys the config keys to delete
   */
  public void deleteConfigKeys(Set<String> deleteKeys) {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeDeleteConfigKeys(new ArrayList<>(deleteKeys));
    }
  }

  private native void nativeDeleteConfigKeys(List<String> deleteKeys);

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

  public Fragment getFragment(int fragmentId) {
    FragmentMetadata metadata = getFragmentNative(fragmentId);
    return new Fragment(this, metadata);
  }

  private native FragmentMetadata getFragmentNative(int fragmentId);

  /**
   * Returns a {@link Tags} instance for performing tag-related operations on the dataset.
   *
   * @return new {@code Tags} instance for dataset tag operations
   * @see Tags
   */
  public Tags tags() {
    return new Tags();
  }

  /**
   * Replace the schema metadata of the dataset.
   *
   * @param metadata the new table metadata
   */
  public void replaceSchemaMetadata(Map<String, String> metadata) {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeReplaceSchemaMetadata(metadata);
    }
  }

  private native void nativeReplaceSchemaMetadata(Map<String, String> metadata);

  /**
   * Replace target field metadata of the dataset. This method won't affect fields not in the map
   *
   * @param fieldMetadataMap field id to metadata map
   */
  public void replaceFieldMetadata(Map<Integer, Map<String, String>> fieldMetadataMap) {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      for (Integer fieldId : fieldMetadataMap.keySet()) {
        Preconditions.checkArgument(fieldId >= 0, "Field id must be greater than 0");
      }
      nativeReplaceFieldMetadata(fieldMetadataMap);
    }
  }

  private native void nativeReplaceFieldMetadata(
      Map<Integer, Map<String, String>> fieldMetadataMap);

  /** Tag operations of the dataset. */
  public class Tags {

    /**
     * Create a new tag for this dataset.
     *
     * @param tag the tag name
     * @param version the version to tag
     */
    public void create(String tag, long version) {
      try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
        Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
        nativeCreateTag(tag, version);
      }
    }

    /**
     * Delete a tag from this dataset.
     *
     * @param tag the tag name
     */
    public void delete(String tag) {
      try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
        Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
        nativeDeleteTag(tag);
      }
    }

    /**
     * Update a tag to a new version for the dataset.
     *
     * @param tag the tag name
     * @param version the version to tag
     */
    public void update(String tag, long version) {
      try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
        Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
        nativeUpdateTag(tag, version);
      }
    }

    /**
     * List all tags of the dataset.
     *
     * @return a list of tags
     */
    public List<Tag> list() {
      try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
        Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
        return nativeListTags();
      }
    }

    /**
     * Get the version of a tag in the dataset.
     *
     * @param tag the tag name
     * @return the version of the tag
     */
    public long getVersion(String tag) {
      try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
        Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
        return nativeGetVersionByTag(tag);
      }
    }
  }

  /**
   * Execute SQL query on the dataset. The underlying SQL engine is DataFusion. Please refer to the
   * DataFusion documentation for supported SQL syntax.
   *
   * @param sql SELECT statement to execute. The default FROM table name is `dataset`, for example:
   *     SELECT * FROM `dataset` LIMIT 10. If FROM table name is a custom value, the {@link
   *     SqlQuery#tableName(String)} should be invoked to set the custom table name.
   * @return a SqlQuery instance.
   */
  public SqlQuery sql(String sql) {
    return new SqlQuery(this, sql);
  }

  /**
   * Merge source data with the existing target data.
   *
   * <p>This will take in the source, merge it with the existing target data, and insert new rows,
   * update existing rows, and delete existing rows.
   *
   * <p>It is important that after merge insert, the current dataset is changed and should be
   * closed. The merged new dataset is contained in the MergeInsertResult.
   *
   * @param mergeInsert merge insert options
   * @param source ArrowArrayStream source data
   * @return MergeInsertResult containing the new merged Dataset.
   */
  public MergeInsertResult mergeInsert(MergeInsertParams mergeInsert, ArrowArrayStream source) {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      MergeInsertResult result = nativeMergeInsert(mergeInsert, source.memoryAddress());

      Dataset newDataset = result.dataset();
      if (selfManagedAllocator) {
        newDataset.allocator = new RootAllocator(Long.MAX_VALUE);
      } else {
        newDataset.allocator = allocator;
      }

      return result;
    }
  }

  private native MergeInsertResult nativeMergeInsert(
      MergeInsertParams mergeInsert, long arrowStreamMemoryAddress);

  private native void nativeCreateTag(String tag, long version);

  private native void nativeDeleteTag(String tag);

  private native void nativeUpdateTag(String tag, long version);

  private native List<Tag> nativeListTags();

  private native long nativeGetVersionByTag(String tag);
}
