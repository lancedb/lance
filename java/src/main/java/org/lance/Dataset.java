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
package org.lance;

import org.lance.cleanup.CleanupPolicy;
import org.lance.cleanup.RemovalStats;
import org.lance.compaction.CompactionOptions;
import org.lance.delta.DatasetDelta;
import org.lance.index.Index;
import org.lance.index.IndexCriteria;
import org.lance.index.IndexDescription;
import org.lance.index.IndexOptions;
import org.lance.index.IndexParams;
import org.lance.index.IndexType;
import org.lance.index.OptimizeOptions;
import org.lance.io.StorageOptionsProvider;
import org.lance.ipc.DataStatistics;
import org.lance.ipc.LanceScanner;
import org.lance.ipc.ScanOptions;
import org.lance.merge.MergeInsertParams;
import org.lance.merge.MergeInsertResult;
import org.lance.namespace.LanceNamespace;
import org.lance.operation.UpdateConfig;
import org.lance.operation.UpdateMap;
import org.lance.schema.ColumnAlteration;
import org.lance.schema.LanceSchema;
import org.lance.schema.SqlExpressions;
import org.lance.util.JsonUtils;

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
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
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
  private Session session;
  private boolean ownsSession = false;

  private final LockManager lockManager = new LockManager();

  private Dataset() {}

  /**
   * Creates a builder for writing a dataset.
   *
   * <p>This builder supports writing datasets either directly to a URI or through a LanceNamespace.
   * Data can be provided via reader() or stream() methods.
   *
   * <p>Example usage with URI and reader:
   *
   * <pre>{@code
   * Dataset dataset = Dataset.write()
   *     .reader(myReader)
   *     .uri("s3://bucket/table.lance")
   *     .mode(WriteMode.CREATE)
   *     .execute();
   * }</pre>
   *
   * <p>Example usage with namespace and empty table:
   *
   * <pre>{@code
   * Dataset dataset = Dataset.write()
   *     .schema(mySchema)
   *     .namespace(myNamespace)
   *     .tableId(Arrays.asList("my_table"))
   *     .mode(WriteMode.CREATE)
   *     .execute();
   * }</pre>
   *
   * @return A new WriteDatasetBuilder instance
   */
  public static WriteDatasetBuilder write() {
    return new WriteDatasetBuilder();
  }

  /**
   * Creates an empty dataset.
   *
   * @param allocator the buffer allocator
   * @param path dataset uri
   * @param schema dataset schema
   * @param params write params
   * @return Dataset
   * @deprecated Use {@link #write()} builder instead. For example: {@code
   *     Dataset.write().allocator(allocator).schema(schema).uri(path)
   *     .mode(WriteMode.CREATE).execute()}
   */
  @Deprecated
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
              params.getDataStorageVersion(),
              params.getEnableV2ManifestPaths(),
              params.getStorageOptions(),
              params.getInitialBases(),
              params.getTargetBases());
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
   * @deprecated Use {@link #write()} builder instead. For example: {@code
   *     Dataset.write().allocator(allocator).stream(stream).uri(path)
   *     .mode(WriteMode.CREATE).execute()}
   */
  @Deprecated
  public static Dataset create(
      BufferAllocator allocator, ArrowArrayStream stream, String path, WriteParams params) {
    return create(allocator, stream, path, params, null);
  }

  /**
   * Create a dataset with given stream and storage options provider.
   *
   * <p>This method supports credential vending through the StorageOptionsProvider interface, which
   * allows for dynamic credential refresh during long-running write operations.
   *
   * @param allocator buffer allocator
   * @param stream arrow stream
   * @param path dataset uri
   * @param params write parameters
   * @param storageOptionsProvider optional provider for dynamic storage options/credentials
   * @return Dataset
   */
  static Dataset create(
      BufferAllocator allocator,
      ArrowArrayStream stream,
      String path,
      WriteParams params,
      StorageOptionsProvider storageOptionsProvider) {
    return create(allocator, stream, path, params, storageOptionsProvider, null, null);
  }

  private static native Dataset createWithFfiSchema(
      long arrowSchemaMemoryAddress,
      String path,
      Optional<Integer> maxRowsPerFile,
      Optional<Integer> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile,
      Optional<String> mode,
      Optional<Boolean> enableStableRowIds,
      Optional<String> dataStorageVersion,
      Optional<Boolean> enableV2ManifestPaths,
      Map<String, String> storageOptions,
      Optional<List<BasePath>> initialBases,
      Optional<List<String>> targetBases);

  private static native Dataset createWithFfiStream(
      long arrowStreamMemoryAddress,
      String path,
      Optional<Integer> maxRowsPerFile,
      Optional<Integer> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile,
      Optional<String> mode,
      Optional<Boolean> enableStableRowIds,
      Optional<String> dataStorageVersion,
      Optional<Boolean> enableV2ManifestPaths,
      Map<String, String> storageOptions,
      Optional<List<BasePath>> initialBases,
      Optional<List<String>> targetBases);

  private static native Dataset createWithFfiStreamAndProvider(
      long arrowStreamMemoryAddress,
      String path,
      Optional<Integer> maxRowsPerFile,
      Optional<Integer> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile,
      Optional<String> mode,
      Optional<Boolean> enableStableRowIds,
      Optional<String> dataStorageVersion,
      Optional<Boolean> enableV2ManifestPaths,
      Map<String, String> storageOptions,
      Optional<StorageOptionsProvider> storageOptionsProvider,
      Optional<List<BasePath>> initialBases,
      Optional<List<String>> targetBases,
      LanceNamespace namespace,
      List<String> tableId);

  /**
   * Creates a dataset with optional namespace support for managed versioning.
   *
   * <p>When a namespace is provided, the commit handler will use the namespace's
   * create_table_version method for version tracking.
   *
   * @param allocator buffer allocator
   * @param stream arrow stream
   * @param path dataset uri
   * @param params write parameters
   * @param storageOptionsProvider optional provider for dynamic storage options/credentials
   * @param namespace optional namespace implementation for managed versioning (can be null)
   * @param tableId optional table identifier within the namespace (can be null)
   * @return Dataset
   */
  static Dataset create(
      BufferAllocator allocator,
      ArrowArrayStream stream,
      String path,
      WriteParams params,
      StorageOptionsProvider storageOptionsProvider,
      LanceNamespace namespace,
      List<String> tableId) {
    Preconditions.checkNotNull(allocator);
    Preconditions.checkNotNull(stream);
    Preconditions.checkNotNull(path);
    Preconditions.checkNotNull(params);
    Dataset dataset =
        createWithFfiStreamAndProvider(
            stream.memoryAddress(),
            path,
            params.getMaxRowsPerFile(),
            params.getMaxRowsPerGroup(),
            params.getMaxBytesPerFile(),
            params.getMode(),
            params.getEnableStableRowIds(),
            params.getDataStorageVersion(),
            params.getEnableV2ManifestPaths(),
            params.getStorageOptions(),
            Optional.ofNullable(storageOptionsProvider),
            params.getInitialBases(),
            params.getTargetBases(),
            namespace,
            tableId);
    dataset.allocator = allocator;
    return dataset;
  }

  /**
   * Open a dataset from the specified path.
   *
   * @param path file path
   * @return Dataset
   * @deprecated Use {@link #open()} builder instead: {@code Dataset.open().uri(path).build()}
   */
  @Deprecated
  public static Dataset open(String path) {
    return open(
        new RootAllocator(Long.MAX_VALUE), true, path, new ReadOptions.Builder().build(), null);
  }

  /**
   * Open a dataset from the specified path.
   *
   * @param path file path
   * @param options the open options
   * @return Dataset
   * @deprecated Use {@link #open()} builder instead: {@code
   *     Dataset.open().uri(path).readOptions(options).build()}
   */
  @Deprecated
  public static Dataset open(String path, ReadOptions options) {
    return open(new RootAllocator(Long.MAX_VALUE), true, path, options, null);
  }

  /**
   * Open a dataset from the specified path.
   *
   * @param path file path
   * @param allocator Arrow buffer allocator
   * @return Dataset
   * @deprecated Use {@link #open()} builder instead: {@code
   *     Dataset.open().allocator(allocator).uri(path).build()}
   */
  @Deprecated
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
   * @deprecated Use {@link #open()} builder instead: {@code
   *     Dataset.open().allocator(allocator).uri(path).readOptions(options).build()}
   */
  @Deprecated
  public static Dataset open(BufferAllocator allocator, String path, ReadOptions options) {
    return open(allocator, false, path, options, null);
  }

  /**
   * Open a dataset from the specified path with additional options.
   *
   * @param path file path
   * @param options the open options
   * @return Dataset
   */
  static Dataset open(
      BufferAllocator allocator,
      boolean selfManagedAllocator,
      String path,
      ReadOptions options,
      Session session) {
    return open(allocator, selfManagedAllocator, path, options, session, null, null);
  }

  /**
   * Open a dataset from the specified path with additional options and namespace commit handler.
   *
   * @param path file path
   * @param options the open options
   * @param namespace the LanceNamespace to use for managed versioning (null if not using namespace)
   * @param tableId table identifier (null if not using namespace)
   * @return Dataset
   */
  static Dataset open(
      BufferAllocator allocator,
      boolean selfManagedAllocator,
      String path,
      ReadOptions options,
      Session session,
      LanceNamespace namespace,
      List<String> tableId) {
    Preconditions.checkNotNull(path);
    Preconditions.checkNotNull(allocator);
    Preconditions.checkNotNull(options);

    Session effectiveSession = session;
    if (effectiveSession == null && options.getSession().isPresent()) {
      effectiveSession = options.getSession().get();
    }
    long sessionHandle = effectiveSession != null ? effectiveSession.getNativeHandle() : 0;

    Dataset dataset =
        openNative(
            path,
            options.getVersion(),
            options.getBlockSize(),
            options.getIndexCacheSizeBytes(),
            options.getMetadataCacheSizeBytes(),
            options.getStorageOptions(),
            options.getSerializedManifest(),
            options.getStorageOptionsProvider(),
            sessionHandle,
            namespace,
            tableId);
    dataset.allocator = allocator;
    dataset.selfManagedAllocator = selfManagedAllocator;
    if (effectiveSession != null) {
      dataset.session = effectiveSession;
    } else {
      dataset.session = Session.fromHandle(dataset.nativeGetSessionHandle());
      dataset.ownsSession = true;
    }
    return dataset;
  }

  private static native Dataset openNative(
      String path,
      Optional<Long> version,
      Optional<Integer> blockSize,
      long indexCacheSize,
      long metadataCacheSizeBytes,
      Map<String, String> storageOptions,
      Optional<ByteBuffer> serializedManifest,
      Optional<StorageOptionsProvider> storageOptionsProvider,
      long sessionHandle,
      LanceNamespace namespace,
      List<String> tableId);

  /**
   * Creates a builder for opening a dataset.
   *
   * <p>This builder supports opening datasets either directly from a URI or from a LanceNamespace.
   *
   * <p>Example usage with URI:
   *
   * <pre>{@code
   * Dataset dataset = Dataset.open()
   *     .uri("s3://bucket/table.lance")
   *     .readOptions(options)
   *     .build();
   * }</pre>
   *
   * <p>Example usage with namespace:
   *
   * <pre>{@code
   * Dataset dataset = Dataset.open()
   *     .namespace(myNamespace)
   *     .tableId(Arrays.asList("my_table"))
   *     .build();
   * }</pre>
   *
   * @return A new OpenDatasetBuilder instance
   */
  public static OpenDatasetBuilder open() {
    return new OpenDatasetBuilder();
  }

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

  /** Package-private setter for allocator, used by {@link CommitBuilder}. */
  void setAllocator(BufferAllocator allocator) {
    this.allocator = allocator;
  }

  /**
   * Create a new transaction builder at current version for the dataset. The dataset itself will
   * not refresh after the transaction committed.
   *
   * @return A new instance of {@link SourcedTransaction.Builder} linked to the opened dataset.
   */
  public SourcedTransaction.Builder newTransactionBuilder() {
    return new SourcedTransaction.Builder(this);
  }

  /**
   * Commit a single transaction and return a new Dataset with the new version. Original dataset
   * version will not be refreshed.
   *
   * @param transaction The transaction to commit
   * @return A new instance of {@link Dataset} linked to committed version.
   */
  public Dataset commitTransaction(Transaction transaction) {
    return commitTransaction(transaction, false, true);
  }

  /**
   * Commit a single transaction and return a new Dataset with the new version. Original dataset
   * version will not be refreshed.
   *
   * @param transaction The transaction to commit
   * @param detached If true, the commit will not be part of the main dataset lineage.
   * @param enableV2ManifestPaths If true, and this is a new dataset, uses the new V2 manifest
   *     paths. These paths provide more efficient opening of datasets with many versions on object
   *     stores. This parameter has no effect if the dataset already exists. To migrate an existing
   *     dataset, instead use the `migrateManifestPathsV2` method. Default is true. WARNING: turning
   *     this on will make the dataset unreadable for older versions of Lance (prior to 0.17.0).
   * @return A new instance of {@link Dataset} linked to committed version.
   */
  public Dataset commitTransaction(
      Transaction transaction, boolean detached, boolean enableV2ManifestPaths) {
    Preconditions.checkNotNull(transaction);
    Dataset dataset =
        new CommitBuilder(this)
            .detached(detached)
            .enableV2ManifestPaths(enableV2ManifestPaths)
            .execute(transaction);
    if (selfManagedAllocator) {
      dataset.allocator = new RootAllocator(Long.MAX_VALUE);
    } else {
      dataset.allocator = allocator;
    }
    return dataset;
  }

  /**
   * Drop a Dataset.
   *
   * @param path The file path of the dataset
   * @param storageOptions Storage options
   */
  public static native void drop(String path, Map<String, String> storageOptions);

  /**
   * Migrate the manifest paths to the new format.
   *
   * <p>This will update the manifest to use the new v2 format for paths.
   *
   * <p>This function is idempotent, and can be run multiple times without changing the state of the
   * object store.
   *
   * <p>DANGER: this should not be run while other concurrent operations are happening. And it
   * should also run until completion before resuming other operations.
   */
  public void migrateManifestPathsV2() {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeMigrateManifestPathsV2();
    }
  }

  private native void nativeMigrateManifestPathsV2();

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
   * Truncate the dataset by deleting all rows. The schema is preserved and a new version is
   * created.
   */
  public void truncateTable() {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeTruncateTable();
    }
  }

  private native void nativeTruncateTable();

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

  /**
   * @return the latest version of the dataset.
   */
  public long latestVersion() {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeGetLatestVersionId();
    }
  }

  private native long nativeGetLatestVersionId();

  /**
   * Get the initial storage options used to open this dataset.
   *
   * <p>This returns the options that were provided when the dataset was opened, without any refresh
   * from the provider. Returns null if no storage options were provided.
   *
   * @return the initial storage options, or null if none were provided
   */
  public Map<String, String> getInitialStorageOptions() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeGetInitialStorageOptions();
    }
  }

  private native Map<String, String> nativeGetInitialStorageOptions();

  /**
   * Get the latest storage options, potentially refreshed from the provider.
   *
   * <p>If a storage options provider was configured and credentials are expiring, this will refresh
   * them.
   *
   * @return the latest storage options (static or refreshed from provider), or null if no storage
   *     options were configured for this dataset
   * @throws RuntimeException if an error occurs while fetching/refreshing options from the provider
   */
  public Map<String, String> getLatestStorageOptions() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeGetLatestStorageOptions();
    }
  }

  private native Map<String, String> nativeGetLatestStorageOptions();

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
      Dataset newDataset = nativeCheckoutVersion(version);
      if (selfManagedAllocator) {
        newDataset.allocator = new RootAllocator(Long.MAX_VALUE);
      } else {
        newDataset.allocator = allocator;
      }
      return newDataset;
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
      Dataset newDataset = nativeCheckoutTag(tag);
      if (selfManagedAllocator) {
        newDataset.allocator = new RootAllocator(Long.MAX_VALUE);
      } else {
        newDataset.allocator = allocator;
      }
      return newDataset;
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
   * Creates a new index on the dataset
   *
   * @param columns the columns to index from
   * @param indexType the index type
   * @param name the name of the created index
   * @param params index params
   * @param replace whether to replace the existing index
   * @return the metadata of the created index
   * @deprecated please use {@link Dataset#createIndex(IndexOptions)} instead.
   */
  @Deprecated
  public Index createIndex(
      List<String> columns,
      IndexType indexType,
      Optional<String> name,
      IndexParams params,
      boolean replace) {
    return createIndex(
        IndexOptions.builder(columns, indexType, params)
            .replace(replace)
            .withIndexName(name.orElse(null))
            .build());
  }

  /**
   * Creates a new index on the dataset.
   *
   * @param options options for building index
   * @return the metadata of the created index
   */
  public Index createIndex(IndexOptions options) {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeCreateIndex(
          options.getColumns(),
          options.getIndexType().getValue(),
          options.getIndexName(),
          options.getIndexParams(),
          options.isReplace(),
          options.isTrain(),
          options.getFragmentIds(),
          options.getIndexUUID(),
          options.getPreprocessedData().map(ArrowArrayStream::memoryAddress));
    }
  }

  private native Index nativeCreateIndex(
      List<String> columns,
      int indexTypeCode,
      Optional<String> name,
      IndexParams params,
      boolean replace,
      boolean train,
      Optional<List<Integer>> fragments,
      Optional<String> indexUUID,
      Optional<Long> arrowStreamMemoryAddress);

  public void mergeIndexMetadata(
      String indexUUID, IndexType indexType, Optional<Integer> batchReadHead) {
    innerMergeIndexMetadata(indexUUID, indexType.getValue(), batchReadHead);
  }

  private native void innerMergeIndexMetadata(
      String indexUUID, int indexType, Optional<Integer> batchReadHead);

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
   * Returns the session associated with this dataset.
   *
   * <p>The session holds runtime state for the dataset, including index and metadata caches. If a
   * session was provided when opening the dataset, that session is returned. Otherwise, a new
   * session was created automatically.
   *
   * <p>The returned session can be used to open other datasets to share caches.
   *
   * @return the session associated with this dataset
   */
  public Session session() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return session;
    }
  }

  private native long nativeGetSessionHandle();

  /**
   * Count rows matching a filter using a specific scalar index. This directly queries the index and
   * counts matching row addresses, which is more efficient than scanning when the index covers the
   * filter column.
   *
   * @param indexName the name of the scalar index to use
   * @param filter the filter expression (e.g., "column = 5")
   * @param fragmentIds optional list of fragment IDs to restrict the count to
   * @return count of matching rows
   */
  public long countIndexedRows(
      String indexName, String filter, Optional<List<Integer>> fragmentIds) {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      Preconditions.checkArgument(
          indexName != null && !indexName.isEmpty(), "indexName cannot be null or empty");
      Preconditions.checkArgument(
          filter != null && !filter.isEmpty(), "filter cannot be null or empty");
      return nativeCountIndexedRows(indexName, filter, fragmentIds);
    }
  }

  private native long nativeCountIndexedRows(
      String indexName, String filter, Optional<List<Integer>> fragmentIds);

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
   * Get the {@link org.lance.schema.LanceSchema} of the dataset with field ids.
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
   * Get the {@link org.lance.Transaction} of the dataset at the current version.
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

  /**
   * Optimize index metadata and segments for this dataset.
   *
   * @param options options controlling index optimization behavior
   */
  public void optimizeIndices(OptimizeOptions options) {
    Preconditions.checkNotNull(options);
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeOptimizeIndices(options);
    }
  }

  private native void nativeOptimizeIndices(OptimizeOptions options);

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
   * Get all indexes with full metadata.
   *
   * @return list of Index objects with complete metadata including index type and fragment coverage
   */
  public List<Index> getIndexes() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeGetIndexes();
    }
  }

  private native List<Index> nativeGetIndexes();

  /**
   * Get statistics for a specific index in JSON form.
   *
   * <p>The JSON structure matches the Rust/Python index_statistics API.
   *
   * @param indexName the name of the index
   * @return JSON string with index statistics
   */
  public Map<String, Object> getIndexStatistics(String indexName) {
    Preconditions.checkArgument(
        indexName != null && !indexName.isEmpty(), "indexName cannot be null or empty");
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      String jsonDesc = nativeGetIndexStatistics(indexName);
      return JsonUtils.fromJson(jsonDesc);
    }
  }

  private native String nativeGetIndexStatistics(String indexName);

  /**
   * Describe indices on this dataset filtered by criteria.
   *
   * @param criteria filter options such as column, name or index capabilities
   * @return list of index descriptions
   */
  public List<IndexDescription> describeIndices(IndexCriteria criteria) {
    Preconditions.checkNotNull(criteria, "criteria cannot be null");
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeDescribeIndices(Optional.of(criteria));
    }
  }

  /**
   * Describe all indices on this dataset.
   *
   * @return list of index descriptions
   */
  public List<IndexDescription> describeIndices() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeDescribeIndices(Optional.empty());
    }
  }

  private native List<IndexDescription> nativeDescribeIndices(Optional<IndexCriteria> criteria);

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
   * @deprecated Use {@link #newTransactionBuilder()} with {@link UpdateConfig} operation instead
   */
  @Deprecated
  public void updateConfig(Map<String, String> tableConfig) {
    UpdateMap configUpdate = UpdateMap.builder().updates(tableConfig).replace(true).build();

    UpdateConfig operation = UpdateConfig.builder().configUpdates(configUpdate).build();

    Dataset newDataset = newTransactionBuilder().operation(operation).build().commit();
    updateToNewDataset(newDataset);
  }

  /**
   * Delete the config keys of the dataset.
   *
   * @param deleteKeys the config keys to delete
   * @deprecated Use {@link #newTransactionBuilder()} with {@link UpdateConfig} operation instead
   */
  @Deprecated
  public void deleteConfigKeys(Set<String> deleteKeys) {
    Map<String, String> deleteMap = new HashMap<>();
    deleteKeys.forEach(key -> deleteMap.put(key, null));
    UpdateMap configUpdate = UpdateMap.builder().updates(deleteMap).replace(false).build();

    UpdateConfig operation = UpdateConfig.builder().configUpdates(configUpdate).build();

    Dataset newDataset = newTransactionBuilder().operation(operation).build().commit();
    updateToNewDataset(newDataset);
  }

  /**
   * Updates the internal state of this dataset to match the provided new dataset. This is used by
   * deprecated void methods that need to update the current dataset instance.
   */
  private void updateToNewDataset(Dataset newDataset) {
    // Close the current handle to avoid resource leak
    close();

    // Replace all internal state with the new dataset
    this.nativeDatasetHandle = newDataset.nativeDatasetHandle;
    this.allocator = newDataset.allocator;
    this.selfManagedAllocator = newDataset.selfManagedAllocator;

    // Prevent the new dataset from closing the handle when it gets GC'd
    newDataset.nativeDatasetHandle = 0;
  }

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
      if (ownsSession && session != null) {
        session.close();
        session = null;
        ownsSession = false;
      }
    }
  }

  /**
   * Native method to release the Lance dataset resources associated with the given handle.
   *
   * @param handle The native handle to the dataset resource.
   */
  private native void releaseNativeDataset(long handle);

  // ===== BlobFile / Blob dataset entry points (JNI) =====
  private native List<BlobFile> nativeTakeBlobs(List<Long> rowIds, String column);

  private native List<BlobFile> nativeTakeBlobsByIndices(List<Long> rowIndices, String column);

  /**
   * Open blob files for given row ids on a blob column. Names and semantics align with Rust/Python.
   *
   * @param rowIds stable row ids (row addresses)
   * @param column blob column name
   * @return list of BlobFile objects
   */
  public List<BlobFile> takeBlobs(List<Long> rowIds, String column) {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      Preconditions.checkArgument(
          rowIds != null && !rowIds.isEmpty(), "rowIds cannot be null or empty");
      Preconditions.checkArgument(
          column != null && !column.isEmpty(), "column cannot be null or empty");
      return nativeTakeBlobs(rowIds, column);
    }
  }

  /**
   * Open blob files for given row indices on a blob column.
   *
   * @param rowIndices row offsets within dataset
   * @param column blob column name
   * @return list of BlobFile objects
   */
  public List<BlobFile> takeBlobsByIndices(List<Long> rowIndices, String column) {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      Preconditions.checkArgument(
          rowIndices != null && !rowIndices.isEmpty(), "rowIndices cannot be null or empty");
      Preconditions.checkArgument(
          column != null && !column.isEmpty(), "column cannot be null or empty");
      return nativeTakeBlobsByIndices(rowIndices, column);
    }
  }

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

  /** Branch operations aligned with Rust's Dataset branch APIs. */
  public Branches branches() {
    return new Branches();
  }

  /**
   * Create a branch at a specified version. The returned Dataset points to the created branch's
   * initial version.
   *
   * @param branch the branch name to create
   * @param ref the reference to create branch from
   * @return a new Dataset of the branch
   */
  public Dataset createBranch(String branch, Ref ref) {
    Preconditions.checkArgument(branch != null && ref != null, "branch and ref cannot be null");
    return innerCreateBranch(branch, ref, Optional.empty());
  }

  /**
   * Create a branch at a specified version. The returned Dataset points to the created branch's
   * initial version.
   *
   * @param branch the branch name to create
   * @param ref the reference to create branch from
   * @param storageOptions the storage options to create branch with
   * @return a new Dataset of the branch
   */
  public Dataset createBranch(String branch, Ref ref, Map<String, String> storageOptions) {
    Preconditions.checkArgument(branch != null && ref != null, "branch and ref cannot be null");
    Preconditions.checkArgument(
        storageOptions != null && !storageOptions.isEmpty(), "storageOptions cannot be null");
    return innerCreateBranch(branch, ref, Optional.of(storageOptions));
  }

  private Dataset innerCreateBranch(
      String branch, Ref ref, Optional<Map<String, String>> storageOptions) {
    Preconditions.checkArgument(branch != null, "Branch cannot be null");
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeCreateBranch(branch, ref, storageOptions);
    }
  }

  /**
   * Checkout using a unified {@link Ref} which can be a tag, the latest version on main/branch or a
   * specified (branch_name, version_number).
   *
   * @param ref the checkout reference
   * @return a new Dataset instance checked out to the specified reference
   */
  public Dataset checkout(Ref ref) {
    Preconditions.checkNotNull(ref);
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      Dataset newDataset = nativeCheckout(ref);
      if (selfManagedAllocator) {
        newDataset.allocator = new RootAllocator(Long.MAX_VALUE);
      } else {
        newDataset.allocator = allocator;
      }
      return newDataset;
    }
  }

  /**
   * Get the table metadata of the dataset.
   *
   * @return the table metadata as a map of key-value pairs
   */
  public Map<String, String> getTableMetadata() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeGetTableMetadata();
    }
  }

  private native Map<String, String> nativeGetTableMetadata();

  /** Tag operations of the dataset. */
  public class Tags {

    /**
     * Create a new tag on main branch. This is left for compatibility. We should use {@link
     * #create(String, Ref)} instead.
     *
     * @param tag the tag name
     * @param versionNumber the version number to tag
     */
    public void create(String tag, long versionNumber) {
      Preconditions.checkArgument(versionNumber > 0, "versionNumber must be greater than 0");
      create(tag, Ref.ofMain(versionNumber));
    }

    /**
     * Create a new tag on a specified branch.
     *
     * @param tag the tag name
     * @param ref the referenced version to tag
     */
    public void create(String tag, Ref ref) {
      Preconditions.checkArgument(tag != null, "Tag name cannot be null");
      Preconditions.checkArgument(ref != null, "ref cannot be null");
      try (LockManager.WriteLock readLock = lockManager.acquireWriteLock()) {
        Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
        nativeCreateTag(tag, ref);
      }
    }

    /**
     * Creates a new tag on the specified branch. This method will be removed in version 2.0.0. Use
     * {@link #create(String, Ref)} instead.
     *
     * @param tag the name of the tag to create
     * @param versionNumber the version number (or commit reference) to associate with the tag
     */
    @Deprecated
    public void create(String tag, long versionNumber, String targetBranch) {
      create(tag, Ref.ofBranch(targetBranch, versionNumber));
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
     * Update a tag to a new version_number on main. This is left for compatibility. We should use
     * {@link #update(String, Ref)} instead.
     *
     * @param tag the tag name
     * @param versionNumber the versionNumber on main.
     */
    public void update(String tag, long versionNumber) {
      Preconditions.checkArgument(versionNumber > 0, "version_number must be greater than 0");
      nativeUpdateTag(tag, Ref.ofMain(versionNumber));
    }

    /**
     * Update a tag to a new reference.
     *
     * @param tag the tag name
     * @param ref the referenced version to tag
     */
    public void update(String tag, Ref ref) {
      Preconditions.checkArgument(tag != null, "tag cannot be null");
      Preconditions.checkArgument(ref != null, "ref cannot be null");
      try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
        Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
        nativeUpdateTag(tag, ref);
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

  /** Branch operations of the dataset. */
  public class Branches {

    /**
     * Delete a branch and its metadata.
     *
     * @param branchName the branch to delete
     */
    public void delete(String branchName) {
      try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
        Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
        nativeDeleteBranch(branchName);
      }
    }

    /**
     * List all branches in this dataset.
     *
     * @return a list of Branch objects
     */
    public List<Branch> list() {
      try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
        Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
        return nativeListBranches();
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
   * Compute the delta between current version and this version.
   *
   * @param comparedAgainst the version to compare the current dataset against
   * @return a DatasetDelta view
   * @throws IllegalArgumentException if mutual exclusivity or completeness rules are violated
   */
  public DatasetDelta delta(long comparedAgainst) {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeBuildDelta(Optional.of(comparedAgainst), Optional.empty(), Optional.empty());
    }
  }

  /**
   * Compute the delta between both {@code beginVersion} (exclusive) and {@code endVersion}
   * (inclusive).
   *
   * @param beginVersion the beginning version (exclusive) for explicit range
   * @param endVersion the ending version (inclusive) for explicit range
   * @return a DatasetDelta view
   * @throws IllegalArgumentException if mutual exclusivity or completeness rules are violated
   */
  public DatasetDelta delta(long beginVersion, long endVersion) {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeBuildDelta(Optional.empty(), Optional.of(beginVersion), Optional.of(endVersion));
    }
  }

  private native DatasetDelta nativeBuildDelta(
      Optional<Long> comparedAgainst, Optional<Long> beginVersion, Optional<Long> endVersion);

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
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
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

  private native void nativeCreateTag(String tag, Ref ref);

  private native void nativeDeleteTag(String tag);

  private native void nativeUpdateTag(String tag, Ref ref);

  private native List<Tag> nativeListTags();

  private native long nativeGetVersionByTag(String tag);

  // ===== Branch native methods =====
  private native Dataset nativeCheckout(Ref ref);

  private native Dataset nativeCreateBranch(
      String branch, Ref ref, Optional<Map<String, String>> storageOptions);

  private native void nativeDeleteBranch(String branch);

  private native List<Branch> nativeListBranches();

  public Dataset shallowClone(String targetPath, Ref ref) {
    return shallowClone(targetPath, ref, null);
  }

  /**
   * Shallow clone the specified tag into a new dataset at the target path.
   *
   * <p>This creates a new dataset that references the data files from the source dataset without
   * copying them. Only metadata is written at the destination.
   *
   * @param targetPath the URI to clone the dataset into
   * @param ref the referred version of the current dataset
   * @param storageOptions Optional object store options for the destination dataset; empty uses
   *     default store parameters
   * @return a new Dataset instance at the target path
   */
  public Dataset shallowClone(String targetPath, Ref ref, Map<String, String> storageOptions) {
    Preconditions.checkArgument(targetPath != null, "Target path can not be null");
    Preconditions.checkArgument(ref != null, "globalVersion can not be null");
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      Dataset newDataset = nativeShallowClone(targetPath, ref, Optional.ofNullable(storageOptions));
      if (selfManagedAllocator) {
        newDataset.allocator = new RootAllocator(Long.MAX_VALUE);
      } else {
        newDataset.allocator = allocator;
      }
      return newDataset;
    }
  }

  private native Dataset nativeShallowClone(
      String targetPath, Ref ref, Optional<Map<String, String>> storageOptions);

  /**
   * Cleanup dataset based on a specified policy.
   *
   * @param policy cleanup policy
   * @return removal stats
   */
  public RemovalStats cleanupWithPolicy(CleanupPolicy policy) {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeCleanupWithPolicy(policy);
    }
  }

  private native RemovalStats nativeCleanupWithPolicy(CleanupPolicy policy);
}
