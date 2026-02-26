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

import org.lance.io.StorageOptionsProvider;
import org.lance.namespace.LanceNamespace;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.util.Preconditions;

import java.util.List;
import java.util.Map;

/**
 * Builder for committing a {@link Transaction} to a Lance dataset.
 *
 * <p>Supports two modes:
 *
 * <ul>
 *   <li><strong>Dataset-based commit</strong>: commits against an existing dataset.
 *   <li><strong>URI-based commit</strong>: creates or updates a dataset at a URI.
 * </ul>
 *
 * <p>Example usage (dataset-based):
 *
 * <pre>{@code
 * try (Transaction txn = new Transaction.Builder()
 *     .readVersion(dataset.version())
 *     .operation(Append.builder().fragments(fragments).build())
 *     .build();
 *     Dataset committed = new CommitBuilder(dataset).execute(txn)) {
 *     // use committed dataset
 * }
 * }</pre>
 *
 * <p>Example usage (URI-based):
 *
 * <pre>{@code
 * try (Transaction txn = new Transaction.Builder()
 *     .operation(Overwrite.builder().fragments(fragments).schema(schema).build())
 *     .build();
 *     Dataset committed = new CommitBuilder(uri, allocator).execute(txn)) {
 *     // use committed dataset
 * }
 * }</pre>
 */
public class CommitBuilder {
  static {
    JniLoader.ensureLoaded();
  }

  private final Dataset dataset;
  private final String uri;
  private final BufferAllocator allocator;

  private Map<String, String> writeParams;
  private StorageOptionsProvider storageOptionsProvider;
  private LanceNamespace namespace;
  private List<String> tableId;
  private boolean enableV2ManifestPaths = true;
  private boolean detached = false;
  private Boolean useStableRowIds;
  private String storageFormat;
  private int maxRetries = 0;
  private boolean skipAutoCleanup = false;

  /**
   * Create a commit builder for committing against an existing dataset.
   *
   * @param dataset the existing dataset to commit against
   */
  public CommitBuilder(Dataset dataset) {
    Preconditions.checkNotNull(dataset, "Dataset must not be null");
    this.dataset = dataset;
    this.uri = null;
    this.allocator = null;
  }

  /**
   * Create a commit builder for creating or updating a dataset at the given URI.
   *
   * @param uri the target URI for the dataset
   * @param allocator the Arrow buffer allocator for schema export
   */
  public CommitBuilder(String uri, BufferAllocator allocator) {
    Preconditions.checkNotNull(uri, "URI must not be null");
    Preconditions.checkNotNull(allocator, "Allocator must not be null");
    this.dataset = null;
    this.uri = uri;
    this.allocator = allocator;
  }

  /**
   * Set write parameters (storage options) for the commit.
   *
   * @param writeParams the write parameters
   * @return this builder instance
   */
  public CommitBuilder writeParams(Map<String, String> writeParams) {
    this.writeParams = writeParams;
    return this;
  }

  /**
   * Set the storage options provider for credential refresh during URI-based commits.
   *
   * @param provider the storage options provider
   * @return this builder instance
   */
  public CommitBuilder storageOptionsProvider(StorageOptionsProvider provider) {
    this.storageOptionsProvider = provider;
    return this;
  }

  /**
   * Set the namespace for managed versioning during URI-based commits.
   *
   * @param namespace the LanceNamespace instance
   * @return this builder instance
   */
  public CommitBuilder namespace(LanceNamespace namespace) {
    this.namespace = namespace;
    return this;
  }

  /**
   * Set the table ID for namespace-based commit handling.
   *
   * @param tableId the table identifier (e.g., ["workspace", "table_name"])
   * @return this builder instance
   */
  public CommitBuilder tableId(List<String> tableId) {
    this.tableId = tableId;
    return this;
  }

  /**
   * Enable or disable v2 manifest paths for new datasets.
   *
   * <p>Defaults to true. V2 manifest paths allow constant-time lookups for the latest manifest on
   * object storage. Warning: enabling this makes the dataset unreadable for Lance versions prior to
   * 0.17.0.
   *
   * @param enable whether to enable v2 manifest paths
   * @return this builder instance
   */
  public CommitBuilder enableV2ManifestPaths(boolean enable) {
    this.enableV2ManifestPaths = enable;
    return this;
  }

  /**
   * Set whether the commit should be detached from the main dataset lineage.
   *
   * @param detached if true, the commit will not be part of the main dataset lineage
   * @return this builder instance
   */
  public CommitBuilder detached(boolean detached) {
    this.detached = detached;
    return this;
  }

  /**
   * Whether to use stable row ids. This makes the {@code _rowid} column stable after compaction,
   * but not updates.
   *
   * <p>This is only used for new datasets. Existing datasets will use their existing setting.
   * Default is false.
   *
   * @param useStableRowIds whether to use stable row ids
   * @return this builder instance
   */
  public CommitBuilder useStableRowIds(boolean useStableRowIds) {
    this.useStableRowIds = useStableRowIds;
    return this;
  }

  /**
   * Set the storage format to use for the dataset.
   *
   * <p>This is only needed when creating a new empty table. If any data files are passed, the
   * storage format will be inferred from the data files. Valid values: "legacy", "v2_0", "stable",
   * "v2_1", "next", "v2_2".
   *
   * @param storageFormat the storage format name
   * @return this builder instance
   */
  public CommitBuilder storageFormat(String storageFormat) {
    this.storageFormat = storageFormat;
    return this;
  }

  /**
   * Set the maximum number of retries for commit operations.
   *
   * <p>If a commit operation fails, it will be retried up to {@code maxRetries} times. Default is
   * 0.
   *
   * @param maxRetries the maximum number of retries
   * @return this builder instance
   */
  public CommitBuilder maxRetries(int maxRetries) {
    this.maxRetries = maxRetries;
    return this;
  }

  /**
   * Set whether to skip automatic cleanup after commit.
   *
   * <p>Default is false.
   *
   * @param skipAutoCleanup if true, skip automatic cleanup
   * @return this builder instance
   */
  public CommitBuilder skipAutoCleanup(boolean skipAutoCleanup) {
    this.skipAutoCleanup = skipAutoCleanup;
    return this;
  }

  /**
   * Execute the commit with the given transaction.
   *
   * <p>The caller is responsible for closing the transaction (via try-with-resources or {@link
   * Transaction#close()}) to release any native resources held by the operation.
   *
   * @param transaction the transaction to commit
   * @return a new Dataset at the committed version
   */
  public Dataset execute(Transaction transaction) {
    Preconditions.checkNotNull(transaction, "Transaction must not be null");
    if (dataset != null) {
      Dataset result =
          nativeCommitToDataset(
              dataset,
              transaction,
              detached,
              enableV2ManifestPaths,
              writeParams,
              useStableRowIds,
              storageFormat,
              maxRetries,
              skipAutoCleanup);
      result.setAllocator(dataset.allocator());
      return result;
    }
    if (uri != null) {
      Dataset result =
          nativeCommitToUri(
              uri,
              transaction,
              detached,
              enableV2ManifestPaths,
              storageOptionsProvider,
              namespace,
              tableId,
              allocator,
              writeParams,
              useStableRowIds,
              storageFormat,
              maxRetries,
              skipAutoCleanup);
      result.setAllocator(allocator);
      return result;
    }
    throw new IllegalStateException("CommitBuilder requires either a dataset or a URI");
  }

  private static native Dataset nativeCommitToDataset(
      Dataset dataset,
      Transaction transaction,
      boolean detached,
      boolean enableV2ManifestPaths,
      Map<String, String> writeParams,
      Boolean useStableRowIds,
      String storageFormat,
      int maxRetries,
      boolean skipAutoCleanup);

  private static native Dataset nativeCommitToUri(
      String uri,
      Transaction transaction,
      boolean detached,
      boolean enableV2ManifestPaths,
      Object storageOptionsProvider,
      Object namespace,
      Object tableId,
      Object allocator,
      Map<String, String> writeParams,
      Boolean useStableRowIds,
      String storageFormat,
      int maxRetries,
      boolean skipAutoCleanup);
}
